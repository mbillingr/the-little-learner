
import ..CommonAbstractions: tensor

struct MyTensor
    elements::AbstractArray
end

is_scalar(obj) = true
is_scalar(obj::MyTensor) = false

is_tensor(obj) = false
is_tensor(obj::MyTensor) = true

CommonAbstractions.tensor(s) = s
CommonAbstractions.tensor(es::AbstractArray) = MyTensor([tensor(e) for e in es])

tlen(es) = length(es)
tlen(t::MyTensor) = length(t.elements)

trank(t) = 0
trank(t::MyTensor) = 1 + trank(tref(t, 0))

tshape(s) = list()
tshape(t::MyTensor) = cons(tlen(t), tshape(tref(t, 0)))

tref(es, i) = es[i+1]  # 0-based indexing
tref(t::MyTensor, i) = t.elements[i+1]  # 0-based indexing

trefs(t::MyTensor, is) = tensor([t.elements[i+1] for i in is])

tmap(f, ts::MyTensor...) = tensor(map(f, map((t) -> t.elements, ts)...))

flatten_2(t::MyTensor) = tensor(cat(map((te) -> te.elements, t.elements)...; dims=1))


Base.foldr(f, x::MyTensor; init) = foldr(f, x.elements; init=init)


function Base.show(io::IO, t::MyTensor)
    fmt(x) = show(io, round(x, digits=2))
    fmt(x::MyTensor) = show(io, x)
    print(io, "[")
    if tlen(t) > 0
        fmt(tref(t, 0))
    end
    for i in 1:tlen(t)-1
        print(io, " ")
        fmt(tref(t, i))
    end
    print(io, "]")
end

function Base.:(==)(t::MyTensor, u::MyTensor)
    if tlen(t) != tlen(u)
        return false
    end

    for i in 0:tlen(t)-1
        if tref(t, i) != tref(u, i)
            return false
        end
    end

    true
end


# frame 183:23
function ext1(func, base_rank)
    function extended(t)
        if of_rank(base_rank, t)
            return func(t)
        else
            return tmap(extended, t)
        end
    end
    extended
end


function ext2(func, base_rank1, base_rank2)
    function extended(t, u)
        if of_ranks(base_rank1, t, base_rank2, u)
            return func(t, u)
        elseif of_rank(base_rank1, t)
            return tmap((eu) -> extended(t, eu), u)
        elseif of_rank(base_rank2, u)
            return tmap((et) -> extended(et, u), t)
        elseif tlen(t) == tlen(u)
            return tmap(extended, t, u)
        elseif rank_gt(t, u)
            return tmap((et) -> extended(et, u), t)
        elseif rank_gt(u, t)
            return tmap((eu) -> extended(t, eu), u)
        else
            error("cannot apply ", func, " to shapes ", tshape(t), " and ", tshape(u))
        end
    end
    extended
end


# frame 53:24

function sum_1(t)
    result = 0
    for i in 0:tlen(t)-1
        result = result + tref(t, i)
    end
    result
end


struct MyTensor
    elements::AbstractArray
end

is_scalar(obj) = true
is_scalar(obj::MyTensor) = false

tensor(s) = s
tensor(es::AbstractArray) = MyTensor([tensor(e) for e in es])

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

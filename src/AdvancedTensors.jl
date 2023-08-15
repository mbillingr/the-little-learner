
import ..CommonAbstractions: tensor
using ..CommonAbstractions

struct MyTensor
    elements::AbstractArray
end

is_scalar(obj) = true
is_scalar(obj::MyTensor) = false

is_tensor(obj) = false
is_tensor(obj::MyTensor) = true

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


# frame 183:23
function ext1_rho(func, base_rank)
    function extended(t)
        if of_rank(base_rank, t)
            return func(t)
        else
            return tmap(extended, t)
        end
    end
    extended
end


function ext2_rho(func, base_rank1, base_rank2)
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

function ext1_nabla(∇_fn, n)
    function extended(t, z)
        if of_rank(n, t)
            ∇_fn(t, z)
        else
            tmap(extended, t, z)
        end
    end
    extended
end

function ext2_nabla(∇_fn, n, m)
    function extended(t, u, z)
        if of_ranks(n, t, m, u)
            ∇_fn(t, u, z)
        elseif of_rank(n, t)
            return desc_u∇(g, t, u, z)
        elseif of_rank(m, u)
            return desc_t∇(g, t, u, z)
        elseif tlen(t) == tlen(u)
            return tmap2(extended, t, u, z)
        elseif rank_gt(t, u)
            return desc_t∇(g, t, u, z)
        elseif rank_gt(u, t)
            return desc_u∇(g, t, u, z)
        else
            error("cannot apply ", func, " to shapes ", tshape(t), " and ", tshape(u))
        end
    end
    extended
end

tmap2(g, t, u, z) =
    build_gt_gu(tlen(t), (i) -> g(tref(t, i), tref(u, i), tref(z, i)))

function build_gt_gu(tn, init)
    gt = Base.zeros(tn)
    gu = Base.zeros(tn)

    for i in 0:tn-1
        (gti, gui) = init(i)
        gt[i] = gti
        gu[i] = gui
    end

    (tensor(gt), tensor(gu))
end

desc_t∇(g, t, u, z) = 
    build_gt_acc_gu(tlen(t), (i)->g(tref(t, i), u, tref(z, i)))

function build_gt_acc_gu(tn, init)
    gt = Base.zeros(tn)
    gu = 0.0

    for i in 0:tn-1
        (gti, gui) = init(i)
        gt[i] = gti
        gu = add_rho(gu, gui)
    end

    (tensor(gt), tensor(gu))
end

desc_u∇(g, t, u, z) = 
    build_gu_acc_gt(tlen(u), (i)->g(t, tref(u, i), tref(z, i)))

function build_gt_acc_gu(un, init)
    gu = Base.zeros(un)
    gt = 0.0

    for i in 0:un-1
        (gti, gui) = init(i)
        gu[i] = gui
        gt = add_rho(gt, gti)
    end

    (tensor(gt), tensor(gu))
end

ext1(f, n) = prim1(
    ext1_rho(ρ_function(f), n),
    ext1_nabla(∇_function(f), n))

ext2(f, n, m) = prim2(
    ext2_rho(ρ_function(f), n, m),
    ext2_nabla(∇_function(f), n, m))

ρ_function(prim) = prim("ρ_function")
∇_function(prim) = prim("∇_function")

one_like(t) = ext1((x) -> 1.0, 0)(t)
add_rho(t, u) = ext2(Base.:+, 0, 0)(t, u)



function sum_1ρ(t)
    result = 0
    for i in 0:tlen(t)-1
        result = add_rho(result, tref(t, i))
    end
    result
end

sum_1∇(t, z) = tmap((t)->z, t)

sum_1(t) = prim1(sum_1ρ, sum_1∇)(t)

module Tensors
export add_rho, correlate, dot_product_1_1, is_scalar, is_tensor, flatten_2, one_like, random_tensor, 
    rank_gt, sum, sum_1, tensor, tlen, tmap, trank, tref, trefs, of_rank, zeroes, zero_tensor
export ext1, ext2, @ext1, @ext2

export ∇, gradient_of, map_star, ∇_once, dual_star, dual

using ..Schemish

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


#Base.foldr(f, x::MyTensor; init) = foldr(f, x.elements; init=init)


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
            return desc_u∇(extended, t, u, z)
        elseif of_rank(m, u)
            return desc_t∇(extended, t, u, z)
        elseif tlen(t) == tlen(u)
            return tmap2(extended, t, u, z)
        elseif rank_gt(t, u)
            return desc_t∇(extended, t, u, z)
        elseif rank_gt(u, t)
            return desc_u∇(extended, t, u, z)
        else
            error("cannot apply ", func, " to shapes ", tshape(t), " and ", tshape(u))
        end
    end
    extended
end

tmap2(g, t, u, z) =
    build_gt_gu(tlen(t), (i) -> g(tref(t, i), tref(u, i), tref(z, i)))

function build_gt_gu(tn, init)
    gt::Array{Any} = [nothing for _ in 1:tn]
    gu::Array{Any} = [nothing for _ in 1:tn]

    for i in 0:tn-1
        (gti, gui) = init(i)
        gt[i+1] = gti
        gu[i+1] = gui
    end

    (tensor(gt), tensor(gu))
end

desc_t∇(g, t, u, z) = 
    build_gt_acc_gu(tlen(t), (i)->g(tref(t, i), u, tref(z, i)))

function build_gt_acc_gu(tn, init)
    gt::Array{Any} = [nothing for _ in 1:tn]
    gu = 0.0

    for i in 0:tn-1
        (gti, gui) = init(i)
        gt[i+1] = gti
        gu = add_rho(gu, gui)
    end

    (tensor(gt), tensor(gu))
end

desc_u∇(g, t, u, z) = 
    build_gu_acc_gt(tlen(u), (i)->g(t, tref(u, i), tref(z, i)))

function build_gu_acc_gt(un, init)
    gu::Array{Any} = [nothing for _ in 1:un]
    gt = 0.0

    for i in 0:un-1
        (gti, gui) = init(i)
        gu[i+1] = gui
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

one_like(t) = ext1_rho((x) -> 1.0, 0)(t)
add_rho(t, u) = ext2_rho(Base.:+, 0, 0)(t, u)


init_tensor(f) =
    (s) ->
        if len(s) == 1
            tensor(f(ref(s, 0)))
        else
            tensor([init_tensor(f)(refr(s, 1)) for _ in 1:ref(s, 0)])
        end

zero_tensor = init_tensor(zeros)
random_tensor(c, v, s) = init_tensor((n) -> randn(n) .* sqrt(v) .+ c)(s)


function of_rank(n, t)
    while true
        if n == 0
            return is_scalar(t)
        elseif is_scalar(t)
            return false
        else
            n -= 1
            t = tref(t, 0)
        end
    end
end

function of_ranks(n, t, m, u)
    if of_rank(n, t)
        of_rank(m, u)
    else
        false
    end
end

function rank_gt(t, u)
    while true
        if is_scalar(t)
            return false
        elseif is_scalar(u)
            return true
        else
            t = tref(t, 0)
            u = tref(u, 0)
        end
    end
end

function correlate(filterbank, signal)
    b = tlen(filterbank)  # number of filters
    n = tlen(signal)  # singal length

    tensor([
        tensor([
            part_corr(
                tref(filterbank, f),
                signal, k)
            for f in 0:b-1])
        for k in 0:n-1])
end

function part_corr(filter, signal, k)
    n = tlen(signal)
    m = tlen(filter)
    m2 = m ÷ 2  # half filter length

    r = 0.0
    for a in 0:m-1
        if k + a - m2 >= 0
            if k + a - m2 < n
                r += dot_corr(tref(filter, a), tref(signal, k - m2 + a))
            end
        end
    end
    r
end

dot_corr(fltd, sigd) =
    Base.sum(fltd.elements .* sigd.elements)

macro ext1(extended, func, base_rank)
    :(function $extended(t)
        ext1($func, $base_rank)(t)
    end)
end

macro ext1(func, base_rank)
    quote
        @ext1 $func $func $base_rank
    end
end

macro ext2(extended, func, base_rank1, base_rank2)
    :(function $extended(t, u)
        ext2($func, $base_rank1, $base_rank2)(t, u)
    end)
end

macro ext2(func, base_rank1, base_rank2)
    quote
        @ext2 $func $func $base_rank1 $base_rank2
    end
end


function ∇(f, θ)
    wrt = map_star(dual_star, θ)
    ∇_once(f(wrt), wrt)
end
gradient_of = ∇


# Declaring it mutable seems like an easy way to ensure that
# each Dual is a different instance.
mutable struct Dual
    real::Any
    link::Any  # todo: these are functions
end

dual(r, k) = Dual(r, k)

is_dual(_) = false
is_dual(::Dual) = true

is_scalar(_) = false
is_scalar(x::Number) = true
is_scalar(d::Dual) = d.real isa Number

is_dual_like(d) = is_scalar(d) || is_tensor(d)
is_dual_like(d::Dual) = true

ρ(s) = s
ρ(d::Dual) = d.real

κ(_) = end_of_chain
κ(d::Dual) = d.link

map_star(f, y) =
    if is_dual_like(y)
        f(y)
    elseif is_list(y)
        tmap((ve) -> map_star(f, ve), y)
    else
        y
    end
map_star(f, ys::List) = map((y) -> map_star(f, y), ys)

dual_star(d) = dual(ρ(d), end_of_chain)

function ∇_once(y, wrt)
    σ = ∇_σ(y, IdDict())
    map_star((d) -> get(σ, d, 0.0), wrt)
end


∇_σ(y, σ) =
    if is_dual_like(y)
        κ(y)(y, one_like(ρ(y)), σ)
    else
        foldr(∇_σ, y; init=σ)
    end

function end_of_chain(d, z, σ)
    g = get(σ, d, 0.0)
    σ[d] = add_rho(z, g)
    σ
end

function constant(d, z, σ)
    end_of_chain(d, z, σ)
end

function prim1(ρ_fn, ∇_fn)
    inner(a) = ρ_fn(a)
    inner(da::Dual) = prim1_dual(ρ_fn, ∇_fn, da)
    inner(daf::String) =
        if daf == "ρ_function"
            ρ_fn
        elseif daf == "∇_function"
            ∇_fn
        else
            error(daf)
        end
    inner
end

prim1_dual(ρ_fn, ∇_fn, da) =
    let ra = ρ(da)
        dual(
            ρ_fn(ra),
            (d, z, σ) -> let ga = ∇_fn(ra, z)
                κ(da)(da, ga, σ)
            end)
    end

function prim2(ρ_fn, ∇_fn)
    inner(a, b) = ρ_fn(a, b)
    inner(da::Dual, db::Dual) = 
        prim2_dual(ρ_fn, ∇_fn, da, db)
    inner(da::Dual, b) = 
        prim2_dual(ρ_fn, ∇_fn, da, b)
    inner(a, db::Dual) = 
        prim2_dual(ρ_fn, ∇_fn, a, db)
    inner(daf::String) = 
        if daf == "ρ_function"
            ρ_fn
        elseif daf == "∇_function"
            ∇_fn
        else
            error(daf)
        end
    inner
end

prim2_dual(ρ_fn, ∇_fn, da, db) =
    let ra = ρ(da), rb = ρ(db)
        dual(
            ρ_fn(ra, rb),
            (d, z, σ) -> begin
                (ga, gb) = ∇_fn(ra, rb, z)
                σ_hat = κ(da)(da, ga, σ)
                κ(db)(db, gb, σ_hat)
            end
        )
    end

comparator(f) = (da, db) -> f(ρ(da), ρ(db))

exp_0 = prim1(Base.exp, (ra, z) -> Base.exp(ra) * z)
log_0 = prim1(Base.log, (ra, z) -> (1 / ra) * z)
sqrt_0 = prim1(Base.sqrt, (ra, z) -> z / sqrt(ra) / 2)
add_0_0 = prim2(Base.:+, (ra, rb, z) -> (z, z))
sub_0_0 = prim2(Base.:-, (ra, rb, z) -> (z, -z))
mul_0_0 = prim2(Base.:*, (ra, rb, z) -> (rb * z, ra * z))
div_0_0 = prim2(Base.:*, (ra, rb, z) -> ((1 / rb) * z, ((-1 * ra) / (rb * rb)) * z))
expt_0_0 = prim2(Base.:^, (ra, rb, z) -> (z * rb * (ra^(rb - 1)), z * (ra^rb) * log(ra)))
lt_0_0 = comparator(Base.:<)
gt_0_0 = comparator(Base.:>)
le_0_0 = comparator(Base.:<=)
ge_0_0 = comparator(Base.:>=)
eq_0_0 = comparator(Base.:(==))

macro overload1(basefn, fn)
    quote
        $basefn(d::Dual) = $fn(d)
    end
end

macro overload2(basefn, fn)
    quote
        $basefn(da::Dual, db::Dual) = $fn(da, db)
        $basefn(a::Number, db::Dual) = $fn(dual(a, constant), db)
        $basefn(da::Dual, b::Number) = $fn(da, dual(b, constant))
    end
end

@overload1 Base.exp exp_0
@overload1 Base.log log_0
@overload1 Base.sqrt sqrt_0
@overload2 Base.:+ add_0_0
@overload2 Base.:- sub_0_0
@overload2 Base.:* mul_0_0
@overload2 Base.:/ div_0_0
@overload2 Base.:^ expt_0_0
@overload2 Base.:< lt_0_0
@overload2 Base.:> gt_0_0
@overload2 Base.:<= le_0_0
@overload2 Base.:>= ge_0_0
@overload2 Base.:(==) eq_0_0



function sum_1ρ(t)
    result = 0
    for i in 0:tlen(t)-1
        result = add_rho(result, tref(t, i))
    end
    result
end

sum_1∇(t, z) = tmap((t)->z, t)

sum_1 = prim1(sum_1ρ, sum_1∇)

sum(d::Dual) = d

dot_product_1_1ρ(w, t) = sum_1ρ(w * t)
dot_product_1_1∇(w, t, z) = (t*z, w*z)
dot_product_1_1 = prim2(dot_product_1_1ρ, dot_product_1_1∇)


# second-tier functions
zeroes(x) = 0
@ext1 zeroes 0

# extend builtins

#@ext1 Base.:- 0
#@ext1 Base.:+ 0
#@ext1 Base.abs 0
@ext1 Base.exp exp_0 0
@ext1 Base.log log_0 0
@ext1 Base.sqrt sqrt_0 0
@ext1 sum sum_1 1

@ext2(Base.:*, mul_0_0, 0, 0)
@ext2(Base.:/, div_0_0, 0, 0)
@ext2(Base.:+, add_0_0, 0, 0)
@ext2(Base.:-, sub_0_0, 0, 0)


end
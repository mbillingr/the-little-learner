module AppendixA

export ∇, gradient_of, map_star

using ..Schemish
using ..Tensors


function ∇(f, θ)
    wrt = map_star(dual_star, θ)
    ∇_once(f(wrt), wrt)
end
gradient_of = ∇


# Declaring it mutable seems like an easy way to ensure that
# each Dual is a different instance.
mutable struct Dual
    real::Float64
    link::Any  # todo: these are functions
end

dual(r, k) = Dual(r, k)

is_dual(_) = false
is_dual(::Dual) = true

is_scalar(_) = false
is_scalar(::Number) = true
is_scalar(::Dual) = true

ρ(s) = s
ρ(d::Dual) = d.real

κ(_) = end_of_chain
κ(d::Dual) = d.link

map_star(f, y) = 
    if is_scalar(y)
        f(y)
    else
        tmap((ve)->map_star(f, ve), y)
    end
map_star(f, ys::List) = map((y) -> map_star(f, y), ys)

dual_star(d) = dual(ρ(d), end_of_chain)

function ∇_once(y, wrt)
    σ = ∇_σ(y, IdDict())
    map_star((d) -> get(σ, d, 0.0), wrt)
end


∇_σ(y, σ) = 
    if is_scalar(y)
        let k = κ(y)
            k(y, 1.0, σ)
        end
    else
        foldr(∇_σ, y; init=σ)
    end

function end_of_chain(d, z, σ)
    g = get(σ, d, 0.0)
    σ[d] = z + g
    σ
end

function constant(d, z, σ)
    end_of_chain(d, z, σ)
end

function prim1(ρ_fn, ∇_fn)
    (da) -> let ra = ρ(da)
        dual(
            ρ_fn(ra),
            (d, z, σ) -> let ga = ∇_fn(ra, z)
                κ(da)(da, ga, σ)
            end
        )
    end
end

function prim2(ρ_fn, ∇_fn)
    (da, db) -> let ra = ρ(da), rb = ρ(db)
        dual(
            ρ_fn(ra, rb),
            (d, z, σ) -> begin
                (ga, gb) = ∇_fn(ra, rb, z)
                σ_hat = κ(da)(da, ga, σ)
                κ(db)(db, gb, σ_hat)
            end
        )
    end
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

end

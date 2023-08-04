module AppendixA

export ∇, gradient_of, map_star

using ..Schemish


function ∇(f, θ)
    wrt = map_star(dual_star, θ)
    ∇_once(f(wrt), wrt)
end


struct Dual
    real::Float64
    link::Float64
end

Scalar = Union{Number,Dual}

dual(r, k) = Dual(r, k)

is_dual(_) = false
is_dual(::Dual) = true

is_scalar(_) = false
is_scalar(::Scalar) = true

ρ(s) = s
ρ(d::Dual) = d.real

κ(_) = end_of_chain
κ(d::Dual) = d.link

map_star(f::Any, ys) = map((y) -> map_star(f, y), ys)
map_star(f::Any, t::Tensor) = tensor([map_star(f, tref(t, i)) for i in 0:tlen(t)-1])
map_star(f::Any, y::Scalar) = f(y)
map_star(f::Any, y::Number) = f(y)  # required because both Tensor and Scalar are supertypes of Number

dual_star(d) = dual(ρ(d), end_of_chain)

end

module Tensors
export add_rho, correlate, is_scalar, is_tensor, flatten_2, one_like, random_tensor, 
    rank_gt, sum, sum_1, tlen, tmap, trank, tref, trefs, of_rank, zeroes, zero_tensor
export ext1, ext2, @ext1, @ext2

using ..CommonAbstractions
using ..Schemish

include("SimpleTensors.jl")
#include("AdvancedTensors.jl")

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
    :(function $extended(t::MyTensor)
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

# second-tier functions
zeroes(x) = 0
@ext1 zeroes 0

# extend builtins

@ext1 Base.:- 0
@ext1 Base.:+ 0
@ext1 Base.abs 0
@ext1 Base.sqrt 0
@ext1 Base.sin 0
@ext1 Base.cos 0
@ext1 Base.tan 0
@ext1 Base.atan 0
@ext1 Base.sum sum_1 1

@ext2(Base.:*, 0, 0)
@ext2(Base.:/, 0, 0)
@ext2(Base.:+, 0, 0)
@ext2(Base.:-, 0, 0)

end
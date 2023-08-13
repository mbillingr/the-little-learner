module Tensors
export correlate, is_scalar, flatten_2, random_tensor, rank_gt, tensor,
    tlen, tmap, trank, tref, trefs, of_rank, zeroes, zero_tensor
export ext1, ext2, @ext1, @ext2

using ..Schemish

include("SimpleTensors.jl")

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

# second-tier functions

@ext1 zeroes ((x) -> 0.0) 0

# extend builtins

@ext1 Base.:- 0
@ext1 Base.:+ 0
@ext1 Base.abs 0
@ext1 Base.sqrt 0
@ext1 Base.sin 0
@ext1 Base.cos 0
@ext1 Base.tan 0
@ext1 Base.atan 0

@ext2(Base.:*, 0, 0)
@ext2(Base.:/, 0, 0)
@ext2(Base.:+, 0, 0)
@ext2(Base.:-, 0, 0)

end
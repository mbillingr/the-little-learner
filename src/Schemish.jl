module Schemish

export sqr
export cons, len, list, ref, snoc
export Tensor
export is_scalar, tensor, tlen, tref
export gradient_of
export ext1, @ext1, @ext2

# various functions
sqr(x) = x * x

# lists
list(members...) = members
cons(m, ms) = (m, ms...)
snoc(ms, m) = (ms..., m)
ref(ms, i) = ms[i+1]  # 0-based indexing
len(ms) = length(ms)


# tensors
struct MyTensor
    elements::AbstractArray
end

Tensor = Union{Number,MyTensor}

is_scalar(obj::Tensor) = false
is_scalar(obj::Number) = true

tensor(s::Number)::Tensor = s
tensor(ts::Tensor...)::Tensor = MyTensor([t for t in ts])
tensor(es::AbstractArray)::Tensor = MyTensor([tensor(e) for e in es])

tlen(es) = length(es)
tlen(t::MyTensor) = length(t.elements)

tref(es, i) = es[i+1]  # 0-based indexing
tref(t::MyTensor, i) = t.elements[i+1]  # 0-based indexing

trank(t::Number) = 0
trank(t::MyTensor) = 1 + trank(tref(t, 0))

function Base.show(io::IO, t::MyTensor)
    print(io, "[")
    if tlen(t) > 0
        show(io, tref(t, 0))
    end
    for i in 1:tlen(t)-1
        print(io, " ")
        show(io, tref(t, i))
    end
    print(io, "]")
end

function Base.:(==)(t::Tensor, u::Tensor)
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

macro ext1(extended, func, base_rank)
    :(function $extended(t::Tensor)
        T = typeof(t)
        if trank(t) > ($base_rank)
            return T([$extended(e) for e in t.elements])
        else
            return $func(t)
        end
    end)
end

macro ext1(func, base_rank)
    quote
        @ext1 $func $func $base_rank
    end
end

function ext1(func, base_rank)
    function extended(t::Tensor)
        T = typeof(t)
        if trank(t) > (base_rank)
            return T([extended(e) for e in t.elements])
        else
            return func(t)
        end
    end
    extended
end

macro ext2(func, base_rank1, base_rank2)
    :(function $func(t::Tensor, u::Tensor)
        T = typeof(t)
        U = typeof(u)
        m = trank(t)
        n = trank(u)
        if m == ($base_rank1) && n == ($base_rank2)
            return $func(t)
        elseif m == ($base_rank1)
            return U([$func(t, ue) for ue in u.elements])
        elseif n == ($base_rank2)
            return T([$func(te, u) for te in t.elements])
        elseif tlen(t) == tlen(u)
            return T([$func(te, ue) for (te, ue) in zip(t.elements, u.elements)])
        elseif n > m
            return U([$func(t, ue) for ue in u.elements])
        elseif m > n
            return T([$func(te, u) for te in t.elements])
        else
            error("I don't think we should ever reach this")
        end
    end)
end

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
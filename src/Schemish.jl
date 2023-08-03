module Schemish

export cons, len, list, ref, snoc
export is_scalar, tensor, tlen, tref

import Base.*, Base.+

# lists
list(members...) = members
cons(m, ms) = (m, ms...)
snoc(ms, m) = (ms..., m)
ref(ms, i) = ms[i+1]  # 0-based indexing
len(ms) = length(ms)


# tensors
struct Tensor
    elements::AbstractArray
end

is_scalar(obj::Number) = true
is_scalar(obj::Tensor) = false
is_scalar(obj::Tuple) = false
is_scalar(obj::AbstractArray) = false

tensor(s::Number) = s
tensor(ts::Tensor...) = Tensor([t for t in ts])
tensor(es::AbstractArray) = Tensor([tensor(e) for e in es])

tlen(es) = length(es)
tlen(t::Tensor) = length(t.elements)

tref(es, i) = es[i+1]  # 0-based indexing
tref(t::Tensor, i) = t.elements[i+1]  # 0-based indexing

function *(ta::Tensor, tb::Tensor)
    return Tensor([a * b for (a, b) in zip(ta.elements, tb.elements)])
end

function *(a::Number, t::Tensor)
    return Tensor([a * b for b in t.elements])
end

function *(t::Tensor, a::Number)
    return Tensor([a * b for b in t.elements])
end

function +(ta::Tensor, tb::Tensor)
    return Tensor([a * b for (a, b) in zip(ta.elements, tb.elements)])
end

function +(a::Number, t::Tensor)
    return Tensor([a + b for b in t.elements])
end

function +(t::Tensor, a::Number)
    return Tensor([a + b for b in t.elements])
end


end
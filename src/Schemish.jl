module Schemish

export cons, len, list, ref, snoc
export is_scalar, tensor, tlen, tref

# lists
list(members...) = members
cons(m, ms) = (m, ms...)
snoc(ms, m) = (ms..., m)
ref(ms, i) = ms[i+1]  # 0-based indexing
len(ms) = length(ms)


# tensors
is_scalar(obj::Number) = true
is_scalar(obj::Tuple) = false
is_scalar(obj::AbstractArray) = false
tensor(elements...) = elements
tlen(es) = length(es)
tref(es, i) = es[i+1]  # 0-based indexing

end
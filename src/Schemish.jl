module Schemish

export sqr
export List, append, cons, len, list, ref, refr, snoc
export is_scalar, flatten_2, rank_gt, tensor, tlen, tmap, trank, tref, trefs, of_rank, zeroes
export gradient_of
export ext1, ext2, @ext1, @ext2
export @with_hypers, @with_hyper

# various functions
sqr(x) = x * x

# lists
List = Tuple
list(members...)::List = members
cons(m, ms::List)::List = (m, ms...)
snoc(ms::List, m)::List = (ms..., m)
ref(ms::List, i) = ms[i+1]  # 0-based indexing
refr(ms::List, i) = ms[i+1:end]
len(ms::List) = length(ms)
append(a::List, b::List) = (a..., b...)


# tensors
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

tref(es, i) = es[i+1]  # 0-based indexing
tref(t::MyTensor, i) = t.elements[i+1]  # 0-based indexing

trefs(t::MyTensor, is) = tensor([t.elements[i+1] for i in is])

tmap(f, ts::MyTensor...) = tensor(map(f, map((t) -> t.elements, ts)...))

flatten_2(t::MyTensor) = tensor(cat(map((te) -> te.elements, t.elements)...; dims=1))

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

Base.foldr(f, x::MyTensor; init) = foldr(f, x.elements; init=init)

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
        T = typeof(t)
        U = typeof(u)
        m = trank(t)
        n = trank(u)
        if of_ranks(base_rank1, t, base_rank2, u)
            return func(t, u)
        elseif of_rank(base_rank1, t)
            return tmap((eu)->extended(t, eu), u)
        elseif of_rank(base_rank2, u)
            return tmap((et)->extended(et, u), t)
        elseif tlen(t) == tlen(u)
            return tmap(extended, t, u)
        elseif rank_gt(t, u)
            return tmap((et)->extended(et, u), t)
        elseif rank_gt(u, t)
            return tmap((eu)->extended(t, eu), u)
        else
            error("I don't think we should ever reach this")
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

# for temporarily setting a hyperparameter (aka dynamic variable)
macro with_hyper(var, val, body)
    backup = gensym()
    esc(
        quote
            global $var
            $backup = $var
            $var = $val
            try
                $body
            finally
                global $var = $backup
            end
        end
    )
end

# for temporarily setting hyperparameters (aka dynamic variables)
macro with_hypers(args...)
    body = args[end]
    vars = args[1:2:end-2]
    vals = args[2:2:end-1]
    for (var, val) in zip(vars, vals)
        body = :(@with_hyper($var, $val, $body))
    end
    esc(body)
end

end
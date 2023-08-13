module Schemish

export sqr
export List, append, cons, len, list, ref, refr, snoc
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
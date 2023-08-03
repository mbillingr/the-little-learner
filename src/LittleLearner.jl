module LittleLearner

export rank, shape

using ..Schemish

# frame 39:37
function shape(t)
    result = list()
    while !is_scalar(t)
        # the iterative implementation requires us to append to the list,
        # in contrast to the book's recursive implementation that prepends.
        result = snoc(result, tlen(t))
        t = tref(t, 0)
    end
    result
end

# frame 42:44
function rank(t)
    result = 0
    while !is_scalar(t)
        # this is mostly equivalent to the books accumulator passing style;
        # we just mutate the accumulator in place.
        result += 1
        t = tref(t, 0)
    end
    result
end

end
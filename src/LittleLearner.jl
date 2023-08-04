module LittleLearner

export rank, shape

using ..Schemish


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
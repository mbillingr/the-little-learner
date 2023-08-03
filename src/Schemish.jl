module Schemish

export list, ref

# 0-based indexing
ref(obj, idx) = obj[idx + 1]

# list construction
list(items...) = items

end
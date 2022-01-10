module utils

function reshape!(s)
    s_ = s[1]
    for i in 2:length(s) s_ = hcat(s_,s[i]) end
    s = transpose(s_)
    #return s
end

function reshape(s)
    s_ = s[1]
    for i in 2:length(s) s_ = hcat(s_,s[i]) end

    return transpose(s_)
end


end# module 
#bound_top = 0
#bound_right = 0
#bound_bottom = 0
#bound_left = 0

bound_top = 1
bound_right = 320
bound_bottom = 478
bound_left = 1

# bound_top = -1
# bound_right = -1
# bound_bottom = -1
# bound_left = -1

def test(top, right, bottom, left):    

    if (top > bound_top) and (top < bound_bottom):

        if (left > bound_left) and (left < bound_right):
        
            return False


    if (bottom > bound_top) and (bottom < bound_bottom):
        
        if (right > bound_left) and (right < bound_right):
        
            return False

    return True


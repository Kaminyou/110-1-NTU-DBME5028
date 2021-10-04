def count_numbers(sorted_list, less_than):
    low = 0
    high = len(sorted_list)
    while high > low:
        mid = (high + low) // 2
        if sorted_list[mid] < less_than: 
            low = mid + 1
        else: 
            high = mid
    return low

if __name__ == "__main__":
    sorted_list = [1, 3, 5, 7]
    print(count_numbers(sorted_list, 4)) # should print 2
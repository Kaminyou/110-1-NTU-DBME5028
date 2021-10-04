def find_two_sum(numbers, target_sum):
    """
    :param numbers: (list of ints) The list of numbers.
    :param target_sum: (int) The required target sum.
    :returns: (a tuple of 2 ints) The indices of the two elements whose sum is equal to target_sum
    """
    previous = {}
    for idx, number in enumerate(numbers):
        need = target_sum - number
        if need in previous:
            return (previous[need], idx)
        else:
            previous[number] = idx

if __name__ == "__main__":
    print(find_two_sum([3, 1, 5, 7, 5, 9], 10))
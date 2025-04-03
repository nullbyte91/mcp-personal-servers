
def find_duplicates(numbers):
    '''
    Find all numbers that appear more than once in the input list.
    This implementation is inefficient and can be optimized.
    '''
    duplicates = []
    
    # Inefficient O(n²) implementation
    for i in range(len(numbers)):
        for j in range(len(numbers)):
            if i != j and numbers[i] == numbers[j] and numbers[i] not in duplicates:
                duplicates.append(numbers[i])
    
    return duplicates

def main():
    # Create a test list with duplicates
    test_list = list(range(1000)) + list(range(500))
    
    # Find duplicates
    result = find_duplicates(test_list)
    print(f"Found {len(result)} duplicates")
    
if __name__ == "__main__":
    main()

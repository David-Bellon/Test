def simpleArraySum(ls):
    suma = 0
    for numbers in ls:
        suma = suma + numbers
    return suma

ar_count = int(input().strip())

ar = list(map(int, input().rstrip().split()))

result = simpleArraySum(ar)


print(result)
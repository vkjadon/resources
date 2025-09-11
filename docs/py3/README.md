## Tuple and List

The common difference between Tuple and the List is that the tuple are immutable and the List is mutable. The tuple data is arranged in paranthesis `()`. Following example will make it more clear. 

```js
# Tuple example (immutable)
my_tuple = (1, 2, "M", 3, 25.7)
# my_tuple[1] = 100               # Uncomment to see the effect
print(my_tuple, my_tuple[0])
```

```js
# List example (mutable)
my_list = [1, 2, "M", 3, 25.7]
# my_list[1] = 100               # Uncomment to see the effect
print(my_list, my_list[0])  
```
The lists are mutable that we can examine by executing the above code after uncommenting the assignment line. So, you should use list when you need flexibility, use tuple when you when you want to store data that should not change such as days of the week, directions, coordinates etc. 

```js
days = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
print(days[0])  # Mon
# days[0] = "Monday" #Not allowed
```
```js
locations = { "Delhi" : (28.7041, 77.1025): , "Mumbai" : (19.0760, 72.8777) }
print(locations["Delhi"])
```

## Reference Copy

We can create a copy of the list by assigning the existing list (list_v1) to new list (list_v2).

```js
list_v1 = [1, 2, 3, 7, 8]

# Let us copy list_v1 to list_v2
list_v2 = list_v1

list_v2.append(4)

list_v1[1]=5
list_v2[2]=6
print(f"Original List {list_v1} , New List {list_v2}")
```
<b>Output:</b>
<div class="note-box">
Original List [1, 5, 6, 7, 8, 4] , New List [1, 5, 6, 7, 8, 4]
</div>
Note that any change in any of the list affects both the lists. This is called the shallow copy where both lists refer to <b>same object</b>. To understand this let us explore the addresses of both the lists

```js
print(f" address of first list : {id(list_v1)}")
print(f" address of second list : {id(list_v2)}")
```
<b>Outout:</b>
<div class="note-box">
 address of first list : 133491606827328

 address of second list : 133491606827328
</div>
Your addresses may be different but you should also get two identical addresses.


## Shallow Copy 

Shallow copy can be created using using `copy` method or `list()` constructor. `list()` is a constructor function that creates and returns a new list object.

```js
list_v2=list(list_v1)
list_v2.append(7)
list_v2[3]=20
print(f"Original List {list_v1} , New List {list_v2}")
print(f"list_v1's address: {id(list_v1)}")
print(f"list_v2's address: {id(list_v2)}")
```
<div class="note-box">
Original List [1, 5, 6, 7, 8, 4] , New List [1, 5, 6, 20, 8, 4, 7]

my_list's address: 133491606827328

another_list's address: 133491606820928
</div>

Note that the addresses in this case are different. You addresses may be different from the output shown here.

```js
list_v2=list_v1.copy()
list_v2.append(7)
list_v2[3]=20
print(f"Original List {list_v1} , New List {list_v2}")
print(f"list_v1's address: {id(list_v1)}")
print(f"list_v2's address: {id(list_v2)}")
```
The out is similar to the above. Again with different addresses.

Now, let us apply `list` constructor on nested list 

```js
list_v1 = [1, 2, [3, 4]]
list_v2 = list(list_v1)

list_v2[2][0] = 99
list_v2[0] = 9

print("Original List:", list_v1)
print("Copy List:", list_v2)
```
The list() constructor creates a shallow copy of the list in which main list is not affected but nested list is affected in both original list and the copy of the list.

## Deep Copy

We can use `deepcopy` method of `copy` module to make all the nested list a different object.

```js
import copy
list_v1 = [1, 2, [3, 4]]
list_v2 = copy.deepcopy(list_v1)

list_v2[2][0] = 99
list_v2[0] = 9

print("Original List:", list_v1)
print("Copy List:", list_v2)
```

<div class="note-box">
Original List: [1, 2, [3, 4]]

Copy List: [9, 2, [99, 4]]
</div>


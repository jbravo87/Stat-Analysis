"""
Created on Sunday Jan 23 2022

@author: J. Bravo
"""
# Import necessary librarie(s)
using Random

#Random.seed!(3); randstring()

# Random Number Generator -> rng
rng = Random.seed!(1729)

N = 41

# A random permutation command experimented with
#randperm(rng, N)

# Setting up some arrays that will serve as point for the cities in the TSP example.
# Call the random number Generator
rng
x = rand(N)
y = rand(N)
points = hcat(x, y)
cities = points
#itinerary = Vector(1:N)
itinerary = 1:N

# Functio that creates collection of lines connecting all the points.
lines = []
#for k in 1:length(itinerary)
#for k in itinerary
#    push!cities(lines, (cities[itinerary[k]], cities[itinerary[k+1]]))
#end
println("Consider the following:")
for i in 1:N-1
    println(cities[itinerary[i]], cities[itinerary[i+1]])
end

points2 = []

for i in 1:N
    push!(points2, (x[i], y[i]))
end

points2

#for k in 1:N-1
#    push!(lines, ([cities[k, k], cities[k, 2]]))
#end

lines2 = []
for j in 1:N-1
    push!(lines2, [[x[j], y[j]], [x[j + 1], y[j + 1]]])
end

lines3 = []
for k in 1:N - 1
    push!(lines3, [[cities[k, 1], cities[k, 2]], [cities[k+1, 1], cities[k+1, 2]]])
end

# Will calculate Euclidean Distance
euclid_dist = sqrt(abs(lines3[1][2][1] - lines3[1][1][1])^2 + abs(lines3[1][2][2] - lines3[1][1][2])^2)
println("The euclidean distance is: ", euclid_dist)

lines4 = []
for i in 1:N-1
    push!(lines4, [[cities[i, 1], cities[i, 2]], [cities[i+1, 1], cities[i+1, 2]]])
end

# For number of rows of a given array
size(cities, 1)
# For number of columns of a given array
size(cities, 2)

# Now want to bundle into a function.
# Will take two arguements.
# An N x 2 matrix for x and y coordinates, and N <- number of cities
function create_lines(cities)
    n = size(cities, 1) - 1
    lines = []
    for k in 1:n
        push!(lines, [[cities[k, 1], cities[k, 2]], [cities[k+1, 1], cities[k+1, 2]]])
    end
    return lines
end
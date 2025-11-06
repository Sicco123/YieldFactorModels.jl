using Lux
using Random 

function main()
    net = Chain(Dense(1 => 3, tanh_fast), Dense(3 => 1; use_bias = false))
    
    # get params and state 
    rng = Random.default_rng()
    params, states = Lux.setup(rng, net)

    inputs = rand(rng, 1, 10)

    random_params = randn(9)
    println("Original params:")
    println(params)
    println(size(params.layer_1.weight))
    
    # Reconstruct params with views into random_params
    params = (
        layer_1 = (
            weight = reshape(view(random_params, 1:3), 3, 1),
            bias = view(random_params, 4:6)
        ),
        layer_2 = (
            weight = reshape(view(random_params, 7:9), 1, 3),
        ),
    )
    println(typeof(states))
    println(states)
    println(typeof(params))
    # create new state 
    
    println("After creating views:")
    println(params.layer_1.weight)
    
    # Now modifying random_params will change params.layer_1.weight
    random_params[1] = 100.0

    println("After modifying random_params[1]:")
    println(params.layer_1.weight)  # This will show 100.0

    # lambda func which maps states into net and gets inputs and params as output through the net
    map_states = (params) -> net(inputs, params, states)
    println(typeof(map_states))
    output, states = map_states( params)

    return output, states
end
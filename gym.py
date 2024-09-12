import numpy as np

from raylib import ffi, rl, colors

# window values
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 900

# maximum number of neurons to show in gym figure
MAX_NEURONS = 25

def init_render() -> None:
    rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, b"NN")
    rl.SetTargetFPS(60)

def close_render() -> None:
    rl.CloseWindow()

def render_fcn(model, step, loss, acc) -> None:

    low_color = colors.RED
    high_color = colors.DARKBLUE

    rl.BeginDrawing()
    rl.ClearBackground(colors.BLACK)
	
    # padding on window borders
    hpad, vpad = int(WINDOW_WIDTH*0.1), int(WINDOW_HEIGHT*0.1)  # 10% of padding
    # redefined nn inside window minus padding
    nn_width = WINDOW_WIDTH - 2*hpad
    nn_height = WINDOW_HEIGHT - 2*vpad
    # x-coordinate options
    layer_width = nn_width / (len(model.nn) + 1)  # +1 as the output layer is not considered in the FCNN
    layer_xcenter = layer_width / 2

    # loop over each layer of the architecture
    for i in range(len(model.nn)):
        nn_layer = model.nn[i]
        # grab number of neurons pero layer
        layer_neurons = nn_layer.weight.shape[1]
        next_layer_neurons = nn_layer.weight.shape[0]
        if layer_neurons > MAX_NEURONS: layer_neurons = MAX_NEURONS
        if next_layer_neurons > MAX_NEURONS: next_layer_neurons = MAX_NEURONS

        # y-coordinate options
        layer_height = nn_height / layer_neurons
        layer_ycenter = layer_height / 2
        next_layer_height = nn_height / next_layer_neurons
        next_layer_ycenter = next_layer_height / 2
        next_layer_height = nn_height / next_layer_neurons
        next_layer_ycenter = next_layer_height / 2

        # over each neuron of the layer
        for j in range(layer_neurons):
            # position of neuron in layer
            x1 = hpad + layer_xcenter + i*layer_width
            y1 = vpad + layer_ycenter + j*layer_height

            # iterate over next layer to draw line: DrawLine(startPosX, startPosY, endPosX, endPosY, color)
            for k in range(next_layer_neurons):
                # position of target neuron
                x2 = hpad + layer_xcenter + (i+1)*layer_width
                y2 = vpad + next_layer_ycenter + k*next_layer_height
                # weight between neuron-j of layer-i and neuron-k of layer-(i+1)
                w_kj = _sigmoid(nn_layer.weight.numpy()[k,j])
                high_color = list(high_color)
                high_color[-1] = int(255 * w_kj)
                rl.DrawLineEx([int(x1), int(y1)], [int(x2), int(y2)], 5, rl.ColorAlphaBlend(low_color, high_color, colors.WHITE))

            # draw neurons of layer
            neuron_radius = 10 + MAX_NEURONS / layer_neurons
            if i == 0:
                rl.DrawCircle(int(x1), int(y1), neuron_radius, colors.DARKBLUE)
            else:
                b_j = _sigmoid(model.nn[i-1].bias.numpy()[j])
                high_color = list(high_color)
                high_color[-1] = int(255 * b_j)
                rl.DrawCircle(int(x1), int(y1), neuron_radius, rl.ColorAlphaBlend(low_color, high_color, colors.WHITE))

        # this is the last layer so manually include the output layer
        next_neuron_radius = 10 + MAX_NEURONS / next_layer_neurons
        if i+1 == len(model.nn):
            output_neurons = nn_layer.weight.shape[0]
            last_layer_height = nn_height / output_neurons
            last_layer_ycenter = last_layer_height / 2
            for l in range(output_neurons):
                x = hpad + layer_xcenter + (i+1)*layer_width
                y = vpad + last_layer_ycenter + l*last_layer_height
                b_j = _sigmoid(nn_layer.bias.numpy()[l])
                high_color = list(high_color)
                high_color[-1] = int(255 * b_j)
                rl.DrawCircle(int(x), int(y), next_neuron_radius, rl.ColorAlphaBlend(low_color, high_color, colors.WHITE))
    
    text = f"[step - {step}] - loss: {loss:4.2f} - acc: {acc:4.2f}"
    rl.DrawText(bytes(text, "utf-8"), WINDOW_WIDTH//4, 25, 22, colors.WHITE)
    rl.EndDrawing()

def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def get_opt_layers(layer_name, batch_size):
    opt_operations = []
    shape = None
    for name, module in model.named_modules():
        if name == layer_name:
            for param in module.parameters():
                opt_operations.append(param)
                shape = param[:batch_size].shape
            break
    return opt_operations, shape


# FDA attack
def get_fda_loss(opt_operations, batch_size):
    loss = 0
    num_layers = len(opt_operations)

    for layer in opt_operations:
        tensor = layer[:batch_size]
        mean_tensor = torch.stack([torch.mean(tensor, -1)] * tensor.shape[-1], -1)
        wts_good = tensor < mean_tensor
        wts_good = wts_good.float()
        wts_bad = tensor >= mean_tensor
        wts_bad = wts_bad.float()

        l2_loss_good = torch.nn.functional.mse_loss(wts_good * layer[batch_size:], torch.zeros_like(layer[batch_size:]))
        l2_loss_bad = torch.nn.functional.mse_loss(wts_bad * layer[batch_size:], torch.zeros_like(layer[batch_size:]))

        loss += torch.log(l2_loss_good / l2_loss_bad)

    loss /= num_layers
    return loss


# Our Loss
def get_main_loss(opt_operations, weights,base_feature):
    loss = 0
    gamma = 1.0
    for layer in opt_operations:
        ori_tensor = layer[:FLAGS.batch_size]
        adv_tensor = layer[FLAGS.batch_size:]
        attribution = (adv_tensor-base_feature)*weights
        #attribution = (adv_tensor)*weights
        blank = tf.zeros_like(attribution)
        positive = tf.where(attribution >= 0, attribution, blank)
        negative = tf.where(attribution < 0, attribution, blank)
        positive = positive
        negative = negative
        balance_attribution = positive + gamma*negative
        loss += tf.reduce_sum(balance_attribution) / tf.cast(tf.size(layer), tf.float32)
        
    loss = loss / len(opt_operations)
    return loss


# Preprocess Input
def preprocess(input_tensor, image_size, image_resize, prob):
    rnd = torch.randint(image_size, image_resize, (1,)).item()
    rescaled = F.interpolate(input_tensor, size=(rnd, rnd), mode='nearest')

    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, (1,)).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, (1,)).item()
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, (pad_left, pad_right, pad_top, pad_bottom), value=0)

    padded = F.interpolate(padded, size=(image_size, image_size), mode='nearest')

    if torch.rand(1) < prob:
        return padded
    else:
        return input_tensor
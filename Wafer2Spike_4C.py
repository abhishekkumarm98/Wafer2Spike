import torch
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli


# Coefficient Gain
Cg = 0.3

        
class CurrentBasedLIF(nn.Module):

    def __init__(self, func_v, pseudo_grad_ops, param):

        """
        args:
        func_v: potential function to produce postsynaptic potential values
        pseudo_grad_ops: pseudo gradient operation
        param: (synaptic current decay, voltage decay, voltage threshold, threshold window)
        """

        super(CurrentBasedLIF, self).__init__()
        self.func_v = func_v
        self.pseudo_grad_ops = pseudo_grad_ops
        self.w_scdecay, self.w_vdecay, self.vth, self.cw = param

    def forward(self, input_data, state):

        """
        args:
        input_data: input spike event from presynaptic neurons
        state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        
        return: output spike, (output spike, current, voltage)
        """

        pre_spike, pre_current, pre_volt = state
        current = self.w_scdecay * pre_current + self.func_v(input_data)
        volt = self.w_vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.cw)
        return output, (output, current, volt)



"""
Defining a custom autograd function for computing the gradient
of Heaviside step function using rectangular function.

"""
class PseudoGradSpike(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, vth, cw):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.cw = cw
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        cw = ctx.cw
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < cw
        return Cg * grad_input * spike_pseudo_grad.float(), None, None




class CurrentBasedLIFWithDropout(nn.Module):

    def __init__(self, func_v, pseudo_grad_ops, param):

        """
        args:
        func_v: potential function to produce postsynaptic potential values
        pseudo_grad_ops: pseudo gradient operation
        param: (synaptic current decay, voltage decay, voltage threshold, threshold window)
        """

        super(CurrentBasedLIFWithDropout, self).__init__()
        self.func_v = func_v
        self.pseudo_grad_ops = pseudo_grad_ops
        self.w_scdecay, self.w_vdecay, self.vth, self.cw = param

    def forward(self, input_data, state, mask, train):

        """
        args:
        input_data: input spike event from presynaptic neurons
        state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        mask: dropout mask
        train: training mode (True or False)
        
        return: output spike, (output spike, current, voltage)
        """

        pre_spike, pre_current, pre_volt = state
        current = self.w_scdecay * pre_current + self.func_v(input_data)
        if train is True:
            current = current * mask
        volt = self.w_vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.cw, mask)
        return output, (output, current, volt)



"""
Defining a custom autograd function for computing the gradient
of Heaviside step function using rectangular function for fully
connected layers with dropout.

"""
class PseudoGradSpikeWithDropout(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input, vth, cw, mask):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.cw = cw
        ctx.mask = mask
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        cw = ctx.cw
        mask = ctx.mask
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < cw
        spike_pseudo_grad[mask==0] = 0
        return Cg * grad_input * spike_pseudo_grad.float(), None, None, None



class Wafer2Spike(nn.Module):

    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):

        """
        args:
        numClasses: Number of Classes
        dropout_fc: Dropout percentage for spiking-based fully connected layers
        spike_ts: Number of timesteps
        device: 'cpu' or 'gpu'
        params: (scdecay, vdecay, vth, cw) ; parameters for each LIF neuron 

        """

        super(Wafer2Spike, self).__init__()
        self.device = device
        self.spike_ts = spike_ts
        self.dropout_fc = dropout_fc
        self.scdecay, self.vdecay, self.vth, self.cw = params
        
        pseudo_grad_ops = PseudoGradSpike.apply
        pseudo_grad_ops_with_dropout = PseudoGradSpikeWithDropout.apply


        # Voltage decay for Convolutional Spike Encoding Layer
        self.conv_spk_enc_w_vdecay = nn.Parameter(torch.ones(1, 64, 30, 30, device=self.device) * self.vdecay)
        
        # Synaptic current decay for Convolutional Spike Encoding Layer
        self.conv_spk_enc_w_scdecay = nn.Parameter(torch.ones(1, 64, 30, 30, device=self.device) * self.scdecay)

        
        # Voltage decay for Spiking-Based Convolutional Layer1
        self.Spk_conv1_w_vdecay = nn.Parameter(torch.ones(1, 64, 30, 30, device=self.device) * self.vdecay)

        # Synaptic current decay for Spiking-Based Convolutional Layer1
        self.Spk_conv1_w_scdecay = nn.Parameter(torch.ones(1, 64, 30, 30, device=self.device) * self.scdecay)


        # Voltage decay for Spiking-Based Convolutional Layer2
        self.Spk_conv2_w_vdecay = nn.Parameter(torch.ones(1, 64, 12, 12, device=self.device) * self.vdecay)

        # Synaptic current decay for Spiking-Based Convolutional Layer2
        self.Spk_conv2_w_scdecay = nn.Parameter(torch.ones(1, 64, 12, 12, device=self.device) * self.scdecay)


        # Voltage decay for Spiking-Based Convolutional Layer3
        self.Spk_conv3_w_vdecay = nn.Parameter(torch.ones(1, 64, 12, 12, device=self.device) * self.vdecay)

        # Synaptic current decay for Spiking-Based Convolutional Layer3
        self.Spk_conv3_w_scdecay = nn.Parameter(torch.ones(1, 64, 12, 12, device=self.device) * self.scdecay)


        # Voltage decay for Spiking-Based Convolutional Layer4
        self.Spk_conv4_w_vdecay = nn.Parameter(torch.ones(1, 64, 3, 3, device=self.device) * self.vdecay)

        # Synaptic current decay for Spiking-Based Convolutional Layer4
        self.Spk_conv4_w_scdecay = nn.Parameter(torch.ones(1, 64, 3, 3, device=self.device) * self.scdecay)


        # Voltage decay for Spiking-Based Fully Connected Layer
        self.Spk_fc_w_vdecay = nn.Parameter(torch.ones(1, 256*9, device=self.device) * self.vdecay)
        # Synaptic current decay for Spiking-Based Fully Connected Layer
        self.Spk_fc_w_scdecay = nn.Parameter(torch.ones(1, 256*9, device=self.device) * self.scdecay)
        
        
        # Time-depenedent weight parameters
        self.w_t = nn.Parameter(torch.ones((self.spike_ts), device=self.device) / self.spike_ts)
        
        self.conv_spk_enc = CurrentBasedLIF(nn.Conv2d(1, 64, (7, 7), stride=1, bias=True), pseudo_grad_ops, 
        [self.conv_spk_enc_w_scdecay, self.conv_spk_enc_w_vdecay, self.vth, self.cw])
        
        self.Spk_conv1 = CurrentBasedLIF(nn.Conv2d(64, 64, (7, 7), stride=1, padding="same", bias=True), pseudo_grad_ops, 
        [self.Spk_conv1_w_scdecay, self.Spk_conv1_w_vdecay, self.vth, self.cw])
        
        self.Spk_conv2 = CurrentBasedLIF(nn.Conv2d(64, 64, (7, 7), stride=2, bias=True), pseudo_grad_ops, 
        [self.Spk_conv2_w_scdecay, self.Spk_conv2_w_vdecay, self.vth, self.cw])
        
        self.Spk_conv3 = CurrentBasedLIF(nn.Conv2d(64, 64, (7, 7), stride=1, bias=True, padding="same"), pseudo_grad_ops, 
        [self.Spk_conv3_w_scdecay, self.Spk_conv3_w_vdecay, self.vth, self.cw])
        
        self.Spk_conv4 = CurrentBasedLIF(nn.Conv2d(64, 64, (7, 7), stride=2, bias=True), pseudo_grad_ops, 
        [self.Spk_conv4_w_scdecay, self.Spk_conv4_w_vdecay, self.vth, self.cw])
        
        self.Spk_fc = CurrentBasedLIFWithDropout(nn.Linear(64*9, 256*9, bias=True), pseudo_grad_ops_with_dropout, 
        [self.Spk_fc_w_scdecay, self.Spk_fc_w_vdecay, self.vth, self.cw])
        
        self.nonSpk_fc = nn.Linear(256*9, numClasses)


    def forward(self, input_data, states):

        """
        args:
        input_data: input wafer maps
        states: list of (init spike, init voltage)
        
        return: Summation of all output spikes stacked together (sum of probabilities) over all time steps
        """

        batch_size = input_data.shape[0]
        output_spikes = []
        
        conv_spk_enc_state, Spk_conv1_state, Spk_conv2_state, Spk_conv3_state, Spk_conv4_state, Spk_fc_state = states[0], states[1], states[2], states[3], states[4], states[5]

        mask_fc = Bernoulli(
            torch.full_like(torch.zeros(batch_size, 256*9, device=torch.device("cuda")), 1 - self.dropout_fc)).sample() / (
                       1 - self.dropout_fc)

 
        for step in range(self.spike_ts):
            
            input_spike = input_data  
            conv_spk_enc_spike, conv_spk_enc_state = self.conv_spk_enc(input_spike, conv_spk_enc_state)
            Spk_conv1_spike, Spk_conv1_state = self.Spk_conv1(conv_spk_enc_spike, Spk_conv1_state)
            Spk_conv2_spike, Spk_conv2_state = self.Spk_conv2(Spk_conv1_spike, Spk_conv2_state)
            Spk_conv3_spike, Spk_conv3_state = self.Spk_conv3(Spk_conv2_spike, Spk_conv3_state)
            Spk_conv4_spike, Spk_conv4_state = self.Spk_conv4(Spk_conv3_spike, Spk_conv4_state)

            # Flattening
            flattened_spike = Spk_conv4_spike.view(batch_size, -1)
            Spk_fc_spike, Spk_fc_state = self.Spk_fc(flattened_spike, Spk_fc_state, mask_fc, self.training)
            nonSpk_fc_output = self.nonSpk_fc(Spk_fc_spike)
            output_spikes += [nonSpk_fc_output * self.w_t[step]]
        
            
        return torch.stack(output_spikes).sum(dim=0)
        
        

class CurrentBasedSNN(nn.Module):

    def __init__(self, numClasses, dropout_fc, spike_ts, device, params):

        """
        args:
        numClasses: Number of Classes
        dropout_fc: Dropout percentage for spiking-based fully connected layers
        spike_ts: Number of timesteps
        device: device
        params: (scdecay, vdecay, vth, cw) ; parameters for each LIF neuron
        """

        super(CurrentBasedSNN, self).__init__()
        self.device = device
        self.wafer2spike = Wafer2Spike(numClasses, dropout_fc, spike_ts, device, params)


    def forward(self, input_data):

        """
        args:
        input_data: input wafer maps
        
        return: Summation of all output spikes stacked together (sum of probabilities) over all time steps
        """

        batch_size = input_data.shape[0]


        # Definiing initial States for each spiking layer initialized with a matrix containing zeros.
        
        # For Convolutional Spike Encoding Layer
        conv_spk_enc_current = torch.zeros(batch_size, 64, 30, 30, device=self.device)
        conv_spk_enc_volt = torch.zeros(batch_size, 64, 30, 30, device=self.device)
        conv_spk_enc_spike = torch.zeros(batch_size, 64, 30, 30, device=self.device)
        conv_spk_enc_state = (conv_spk_enc_spike, conv_spk_enc_current, conv_spk_enc_volt)
        
        # For Spiking-Based Convolutional Layer1
        Spk_conv1_current = torch.zeros(batch_size, 64, 30, 30, device=self.device)
        Spk_conv1_volt = torch.zeros(batch_size, 64, 30, 30, device=self.device)
        Spk_conv1_spike = torch.zeros(batch_size, 64, 30, 30, device=self.device)
        Spk_conv1_state = (Spk_conv1_spike, Spk_conv1_current, Spk_conv1_volt)
        
        # For Spiking-Based Convolutional Layer2
        Spk_conv2_current = torch.zeros(batch_size, 64, 12, 12, device=self.device)
        Spk_conv2_volt = torch.zeros(batch_size, 64, 12, 12, device=self.device)
        Spk_conv2_spike = torch.zeros(batch_size, 64, 12, 12, device=self.device)
        Spk_conv2_state = (Spk_conv2_spike, Spk_conv2_current, Spk_conv2_volt)
        
        # For Spiking-Based Convolutional Layer3
        Spk_conv3_current = torch.zeros(batch_size, 64, 12, 12, device=self.device)
        Spk_conv3_volt = torch.zeros(batch_size, 64, 12, 12, device=self.device)
        Spk_conv3_spike = torch.zeros(batch_size, 64, 12, 12, device=self.device)
        Spk_conv3_state = (Spk_conv3_spike, Spk_conv3_current, Spk_conv3_volt)
        
        # For Spiking-Based Convolutional Layer4
        Spk_conv4_current = torch.zeros(batch_size, 64, 3, 3, device=self.device)
        Spk_conv4_volt = torch.zeros(batch_size, 64, 3, 3, device=self.device)
        Spk_conv4_spike = torch.zeros(batch_size, 64, 3, 3, device=self.device)
        Spk_conv4_state = (Spk_conv4_spike, Spk_conv4_current, Spk_conv4_volt)
        
        # For Spiking-Based Fully Connected Layer
        Spk_fc_current = torch.zeros(batch_size, 256*9, device=self.device)
        Spk_fc_volt = torch.zeros(batch_size, 256*9, device=self.device)
        Spk_fc_spike = torch.zeros(batch_size, 256*9, device=self.device)
        Spk_fc_state = (Spk_fc_spike, Spk_fc_current, Spk_fc_volt)
        
        states = (conv_spk_enc_state, Spk_conv1_state, Spk_conv2_state, Spk_conv3_state, Spk_conv4_state, Spk_fc_state)
        
        output = self.wafer2spike(input_data, states)
        
        return output

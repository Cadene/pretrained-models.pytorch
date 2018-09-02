require 'cutorch'
require 'cunn'
require 'cudnn'
require 'image'
vision=require 'torchnet-vision'

net=vision.models.resnet.load{filename='data/resnet152/net.t7',length=152}
print(net)

require 'nn'
nn.Module.parameters = function(self)
   if self.weight and self.bias and self.running_mean and self.running_var then
      return {self.weight, self.bias, self.running_mean, self.running_var}, {self.gradWeight, self.gradBias}

   elseif self.weight and self.bias then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
   elseif self.weight then
      return {self.weight}, {self.gradWeight}
   elseif self.bias then
      return {self.bias}, {self.gradBias}
   else
      return
   end
end

netparams, _ = net:parameters()
print(#netparams)
torch.save('data/resnet152/netparams.t7', netparams)

net=net:cuda()
net:evaluate()
--p, gp = net:getParameters()
input = torch.ones(1,3,224,224)
input[{1,1,1,1}] = -1
input[1] = image.load('data/cat_224.png')
print(input:sum())
input = input:cuda()
output=net:forward(input)

for i=1, 11 do
    torch.save('data/resnet152/output'..i..'.t7', net:get(i).output:float())
end

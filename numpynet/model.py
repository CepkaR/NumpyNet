import numpy as np

class Sequential:
    def __init__(self):
        self.layers = []
        self.param = {}
        self.gradient = {}
        
    def compile(self,loss=None, 
                optim_package={"optim":None,"optim_param":{},"lr_scheduler":None}):
         
        self.loss = loss
        self.optim_dict = {}
        for name in self.param.keys():
            self.optim_dict[name] = optim_package["optim"]().set_params(**optim_package["optim_param"])
            
        self.lr_scheduler = optim_package["lr_scheduler"]
        self.new_lr = optim_package["optim_param"]["lr"]
        if self.lr_scheduler:
            self.lr_scheduler.lr = self.new_lr 
        return self
    
    def add(self,layer):
        layer.name += f"_{len(self.layers)}"
        self.layers.append(layer)
        if layer._param_bool:
            self.param[layer.name] = (layer.W,layer.b)
        
    
    def predict(self,X):
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def _update_param(self):
        for layer in self.layers:
            if layer._param_bool:
                W,b = self.param[layer.name] #layer.W, layer.b
                dW,db = self.gradient[layer.name]
                optim_i =  self.optim_dict[layer.name]
                if self.lr_scheduler:
                    optim_i.lr = self.new_lr
                    
                W_new,b_new = optim_i.update(W,b,dW,db)
                
                self.param[layer.name] = W_new,b_new
                layer.W, layer.b = W_new,b_new
            
    def _propagate_error(self,err):
        if self.loss.name=="Crossentropy":
            #skip last softmax function
            for layer in self.layers[::-1][1:]:
                err,W_er,b_er = layer.backward(err)
                if layer._param_bool:
                    self.gradient[layer.name] = (W_er,b_er)

        else:
            for layer in self.layers[::-1]:
                err,W_er,b_er = layer.backward(err)
                if layer._param_bool:
                    self.gradient[layer.name] = (W_er,b_er)
            
    
    def train(self, train_dataloader=None, val_dataloader=None, epochs=0, metric=lambda x,y: 0, verbose = False):
        self.metric_loss_dict = {"loss_t":[],"metric_t":[],"loss_v":[],"metric_v":[]}
        for epoch in range(epochs):
            metric_t = 0
            loss_t = 0
            for x, y_true in train_dataloader:
                y_pred = self.predict(x)
                metric_t += metric(y_true,y_pred)
                
            
                loss_t += self.loss(y_true,y_pred)
                err = self.loss.backward(y_true,y_pred)
                self._propagate_error(err)
                self._update_param()
                
            loss_t = loss_t / len(train_dataloader)
            self.metric_loss_dict["loss_t"].append(loss_t) 
            
            metric_t = metric_t / len(train_dataloader)
            self.metric_loss_dict["metric_t"].append(metric_t)
            
            metric_v = 0
            loss_v = 0
            for x, y_true in val_dataloader: 
                y_pred = self.predict(x)
                
                metric_v += metric(y_true,y_pred)
                
                loss_v += self.loss(y_true,y_pred)
            
            loss_v = loss_v / len(val_dataloader)
            self.metric_loss_dict["loss_v"].append(loss_v) 
            metric_v = metric_v / len(val_dataloader)
            self.metric_loss_dict["metric_v"].append(metric_v)
               
           
            if verbose:
                print("epoch : {}/{}, train_loss = {:.6f}, train_metric = {:.6f}, val_loss = {:.6f}, val_metric = {:.6f}, lr = {}".format(
                    epoch + 1, epochs, loss_t, metric_t, loss_v, metric_v, self.new_lr)) 
              
            if self.lr_scheduler:
                self.new_lr = self.lr_scheduler.step(loss_v)   
                if self.new_lr == "stop":
                    break
                    
            

import random
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import gurobipy as gp
from gurobipy import GRB
from tensorflow.keras import layers, models
import tensorflow as tf

class GenData:
    """
    This class generates data for a number of objects and a capacity.
    The user can choose what type of correlation is between the weight
    and the value of the object, and how many observations -- i.e. how
    many knapsack problems -- are generated.

    Input: 
        numObject       : positive integer, number of objects.
        Capacity        : positive integer, weight capacity of knapsack.
        correlation     : string, choice out of 'strong','inv_strong','alm_strong',
                          'weak' and 'un'.
        numObservations : integer >= 1, how many knapsack problems should be 
                          generated.

    ==================================================================
    Example:
        data = GenData(n,W,correlation='inv_strong', numObservations=100)
        x_test = data.generate()
    ==================================================================
    """
    
    def __init__(self, numObjects, Capacity, correlation= 'strong', numObservations=1):
        self.numObjects = numObjects
        self.Capacity = Capacity
        self.R = Capacity
        self.correlation = correlation
        self.numObservations = numObservations

    def strong_corr(self):
        w = np.random.randint(1,self.R+1,size=self.numObjects)
        v = w + self.R/10
            
        return w,v

    def inv_strong_corr(self):
        v = np.random.randint(1,self.R+1,size=self.numObjects)
        w = v + self.R/10
            
        return w,v

    def alm_strong_corr(self):
        w = np.random.randint(1,self.R+1,size=self.numObjects)
        v = np.zeros_like(w)
        
        for i in range(self.numObjects):
            low = w[i] + self.R/10 - self.R/500
            high = w[i] + self.R/10 + self.R/500
            v[i] = np.random.randint(np.floor(low), np.ceil(high))
            
        return w,v

    def weak_corr(self):
        w = np.random.randint(1,self.R+1,size=self.numObjects)
        v = np.zeros_like(w)
        
        for i in range(self.numObjects):
            v[i] = np.random.randint(np.maximum(1,w[i] - self.R/10), w[i] + self.R/10)
            
        return w,v

    def un_corr(self):
        w = np.random.randint(1,self.R+1,size=self.numObjects)
        v = np.random.randint(1,self.R+1,size=self.numObjects)
            
        return w,v
    
    def generate(self):
        """
        generate is a fuction which generate the data following some
        correlation, which is initiated by calling the class. 

        Output:
            wv_arr : 3d array containing values and weights of N
                     observations.
            w      : 1d array containing weights.
            v      : 1d array containing values.
        """
        data_corr = {'strong': self.strong_corr,
                     'inv_strong': self.inv_strong_corr,
                     'alm_strong': self.alm_strong_corr,
                     'weak': self.weak_corr,
                     'un': self.un_corr}

        if self.numObservations == 1:
            w,v = data_corr[self.correlation]()
            return w.astype(int),v.astype(int)
        elif self.numObservations > 1:
            wv_arr = []

            for _ in range(self.numObservations):
                arr = np.zeros((self.numObjects,2))
                w,v = data_corr[self.correlation]()
                arr[:,0] = v
                arr[:,1] = w
                wv_arr.append(arr)

            return np.array(wv_arr).astype(int)
        else:
            raise ValueError("Unsupported number of observations: numObservations has to be an integer greater than 0.")
            
        
class KnapsackSolver:
    """
    This class solves the knapsack problem using one of three approaches:
    dynamic programming (DP), binary programming (Bin) or greedy 
    heuristic (GH). One has to initialize the class by stating the 
    solving method, which is by default DP. Then one must call
    the knapsackSolver function with the right inputs to generate the 
    index solution or the function to obtain only the DP table 
    (knapsack_DP_table or knapsack_DP_Ntable for N tables) or the
    backtracking function (knapsack_DP_backtrack)

    ==================================================================
    Example 1:
        solve = KnapsackSolver('GH')
        idx_sol = solve.knapsackSolver(n,W,v,w)

    Example 2:
        solve = KnapsackSolver()
        Ntable = solve.knapsack_DP_Ntable(n,W,ValsWeights_arr)

    Example 2:
        solve = KnapsackSolver()
        table = solve.knapsack_DP_table(n,W,v,w)
        idx_sol = solve.knapsack_DP_backtrack(table, n, W, v, w)
    ==================================================================
    """
    
    def __init__(self, solver='DP'):
        self.solver = solver

    def knapsackSolver(self, numObjects, Capacity, objectValues, objectWeigts):
        """
        knapsackSolver uses the specified solver method to solve the  
        knapsack problem, given number of objects (numObject), knapsack 
        capacity (Capacity), the values of the objects (objectValues) 
        and the weights of the objects (objectWeights). 
        
        Input:
            numObject     : positive integer, number of objects
            Capacity      : positive integer, maximum capacity
            objectValues  : array of size numObject, the values of the 
                            objects
            objectWeights : array of size numObject, the weights of the 
                            objects

        Output:
            indexSol      : 1d array, the indices of the weights/values
                            array which are the solution to the knapsack
                            problem
        """
        
        solver_dict = {'DP': self.knapsack_DP,
                       'Bin': self.knapsack_Bin,
                        'GH': self.knapsack_GH
                        }
            
        return solver_dict[self.solver](numObjects, Capacity, objectValues, objectWeigts)
        
    def knapsack_DP(self, numObjects, Capacity, objectValues, objectWeights):
        """
        knapsack_DP solves the knapsack problem using integer 
        programming, given number of objects (numObject), knapsack 
        capacity (Capacity), the values of the objects (objectValues) 
        and the weights of the objects (objectWeights). It first 
        creates the table using the knapsack_DP_table function and
        then obtains the index solution indexSol by calling the 
        backtrack function knapsack_DP_backtrack. 

        Input:
            numObject     : positive integer, number of objects
            Capacity      : positive integer, maximum capacity
            objectValues  : array of size numObject, the values of the 
                            objects
            objectWeights : array of size numObject, the weights of the 
                            objects

        Output:
            indexSol      : 1d array, the indices of the weights/values
                            array which are the solution to the knapsack
                            problem
        """
        
        M = self.knapsack_DP_table(numObjects, Capacity, objectValues, objectWeights)

        indexSol = self.knapsack_DP_backtrack(M, numObjects, Capacity, objectValues, objectWeights)
        
        return indexSol

    def knapsack_Bin(self, numObjects, Capacity, objectValues, objectWeights):
        """
        knapsack_Bin solves the knapsack problem using integer 
        programming, given number of objects (numObject), knapsack 
        capacity (Capacity), the values of the objects (objectValues) 
        and the weights of the objects (objectWeights).

        After definining model m, variable x is added, which is an
        array type variable with elements having the value 1 (choose
        to put in knapsack) or 0 (choose not to put in knapsack). Then
        create the objective function which is
                    max{ objectValues @ x }
        and add constraint
                    objectWeights @ x <= Capacity
        After the objective function and constraint are added, then the
        model is optimized and a solution is returned.

        Input:
            numObject     : positive integer, number of objects
            Capacity      : positive integer, maximum capacity
            objectValues  : array of size numObject, the values of the 
                            objects
            objectWeights : array of size numObject, the weights of the 
                            objects

        Output:
            indexSol      : 1d array, the indices of the weights/values
                            array which are the solution to the knapsack
                            problem
        """

        # Define model, using matrix1 to use numpy array objects in 
        # objective function and constraint.
        m = gp.Model("matrix1")

        # Create variable
        x = m.addMVar(shape=numObjects, vtype=GRB.BINARY, name="x")

        # Set objective
        m.setObjective(objectValues @ x, GRB.MAXIMIZE)

        # Add constraint
        m.addConstr(objectWeights @ x <= Capacity, name="weightConstraint")

        # Optimize model
        m.optimize()

        # Assign solution
        sol = x.X

        # Assign the array of indices which are included in the knapsack
        indexSol = np.argwhere(sol==1)
        
        return indexSol.reshape(indexSol.size)
    
    def knapsack_GH(self, numObjects, Capacity, objectValues, objectWeights):
        """
        knapsackSolver solves the knapsack problem using a greedy 
        heuristic, given number of objects (numObject), knapsack 
        capacity (Capacity), the values of the objects (objectValues) 
        and the weights of the objects (objectWeights).

        re-peatedly add the item with highest value to weight ratio 
        that fits the unused capacity.

        Input:
            numObject     : positive integer, number of objects
            Capacity      : positive integer, maximum capacity
            objectValues  : array of size numObject, the values of the 
                            objects
            objectWeights : array of size numObject, the weights of the 
                            objects

        Output:
            indexSol      : 1d array, the indices of the weights/values
                            array which are the solution to the knapsack
                            problem
        """
        # create index solution list (indexSol)
        indexSol = []

        # calculate ratio between object values and object weights
        ratio = objectValues / objectWeights

        # define remaining weight to use in while loop
        remaining_w = Capacity

        # while the remaining weight is greater than zero you'll stay in
        # the while loop, but a break is put in for safety
        while remaining_w > 0:
            # define the maximum ratio
            max_ratio = np.max(ratio)

            # if the maximum ratio is zero, that means that all ratios
            # have been put to zero and you want to break the loop
            if max_ratio == 0.0:
                break

            # identify what index the largest ratio has. Note that we
            # have taken the [0,0] element, because numpy's argwhere 
            # return a nested array, e.g. np.array([[1]]), so taking the
            # [0,0] element, just gives the number
            idx = np.argwhere(ratio == max_ratio)[0,0]

            # check if the weight of the object of the largest ratio
            # has is less than or equal to the remaining weight,
            # because this is a constraint of the problem
            if remaining_w >= objectWeights[idx]:
                remaining_w -= objectWeights[idx] # substract weight from remaining weight        
                indexSol.append(idx) # append indexSol with idx
                ratio[idx] = 0 # set ratio of largest ratio to zero
            else:
                ratio[idx] = 0 # set ratio of largest ratio to zero
        
        
        return np.array(indexSol)

    def knapsack_DP_table(self, numObjects, Capacity, objectValues, objectWeights):
        """
        knapsack_DP_table creates the DP table given a number of objects
        (numObjects), capacity of the knapsack (Capacity), values of the
        objects (objectValues) and weight of the objects (objectWeight).
        It does this by first creating a zero array and then filling the
        values of the array according to the optimization procedure of 
        dynamic programming. 

        Input:
            numObject     : positive integer, number of objects
            Capacity      : positive integer, maximum capacity
            objectValues  : array of size numObject, the values of the 
                            objects
            objectWeights : array of size numObject, the weights of the 
                            objects

        Output:
            M             : 2d array, of shape n+1 by W+1 which is the DP
                            table of the specific problem.
        """
        
        # Define 2D array M which holds the solutions to the optimization 
        # problem and list selected_items which will hold the indices of 
        # the selected items which go into the knapsack
        M = np.zeros((numObjects+1,Capacity+1))

        # For loop over i and j which will fill the matrix M with the
        # solutions to the optimization problem, i.e. the maximum value
        # which can be created given a capacity and number of objects.
        for i in range(1,numObjects+1):
            for weight in range(0,Capacity+1):
                if objectWeights[i-1] > weight:
                    M[i,weight] = M[i-1,weight]
                else:
                    M[i,weight] = np.maximum(M[i-1,weight], objectValues[i-1] + M[i-1, weight-objectWeights[i-1]])
        
        return M

    def knapsack_DP_backtrack(self, M, numObjects, Capacity, objectValues, objectWeights):
        """
        knapsack_DP_backtrack backtracks which indices are responsible
        for the largest value in the table. The procedure is also in 
        the comments below. 
        
        Input:
            numObject     : positive integer, number of objects
            Capacity      : positive integer, maximum capacity
            objectValues  : array of size numObject, the values of the 
                            objects
            objectWeights : array of size numObject, the weights of the 
                            objects

        Output:
            indexSol      : 1d array, the indices of the weights/values
                            array which are the solution to the knapsack
                            problem
        """     
        # Start from the bottom-right corner of the matrix, which holds 
        # the maximum value. Trace back through the matrix to find out 
        # which items were included to achieve this maximum value. If the
        # value at the current cell differs from the value in the row 
        # above, it indicates that the current item was included. 
        # Subtract the weight of the included item from the remaining 
        # capacity and move up in the matrix. Continue this process until 
        # you've traced back to the top of the matrix
        selected_items = []
        i, j = numObjects, Capacity

        while i > 0 and j > 0:
            if M[i, j] != M[i-1, j]:
                selected_items.append(i-1)
                j -= objectWeights[i-1]
            i -= 1
        
        indexSol = np.array(selected_items)
        return indexSol[::-1]

    def knapsack_DP_Ntable(self, numObjects, Capacity, ValsWeights):
        """
        knapsack_DP_Ntable creates N (data_num) of tables for the
        given number of objects, capacity and values and weights,
        by calling the knapsack_DP_table N times and appending it
        to a list which turn into an array.

        Input:
            numObject   : positive integer, number of objects
            Capacity    : positive integer, maximum capacity
            ValsWeights : 3d array of shape (data_num, n, 2) 
                            containing the values in the first 
                            column and the weights in the second 
                            column, i.e. the first measurement's 
                            values and weights are in ValsWeights[0]
                            which is of size (n,2) in which the 
                            first column reside the values and in 
                            the second column reside the weights. 

        Output:
            M           : 3d array, of shape (data_num,n+1, W+1) which is the DP
                          table of the specific problem.
        """
        
        Ntable = []
        data_num = np.shape(ValsWeights)[0]

        for i in range(data_num):
            DP_tab = self.knapsack_DP_table(numObjects,Capacity,ValsWeights[i][:,0],ValsWeights[i][:,1])
    
            Ntable.append(DP_tab)

        return np.array(Ntable)


class NeuralNetworkDP:
    """
    The class NeuralNetworkDP helps to create our model without
    the user having to create the architecture themselves. The 
    class is initiate with NeuralNetworkDP(numObjects, Capacity).
    After creating the instance of the class, the user has to 
    create the model by calling the function createModel inside 
    the class, which takes no arguments, and creates the model 
    for you without the user having to specify the architecture.

    ===========================================================
    Code example:

    init_model = NeuralNetworkDP(10,100)
    model = init_model.createModel()
    history = model.fit(x_train, y_train_new, batch_size=32, 
                        epochs=20, validation_split=0.2)
    ===========================================================
    """
    def __init__(self, numObjects, Capacity):
        self.numObjects = numObjects
        self.Capacity = Capacity
    
    def createModel(self):
        # Model Architecture
        model = models.Sequential()

        # Flatten the input (n, 2) to a single vector and 
        # 3 hidden Dense layers and one output layer
        model.add(layers.Flatten(input_shape=(self.numObjects, 2)))
        model.add(layers.Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.1)))
        model.add(layers.Dense(128, activation=tf.keras.layers.LeakyReLU(negative_slope=0.1)))
        model.add(layers.Dense(256, activation=tf.keras.layers.LeakyReLU(negative_slope=0.1)))
        model.add(layers.Dense((self.numObjects + 1) * (self.Capacity + 1) +1))

        model.compile(optimizer='adam', loss='mse')

        # Model summary to visualize the architecture
        model.summary()

        return model

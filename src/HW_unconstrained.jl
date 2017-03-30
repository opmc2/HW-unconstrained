

module HW_unconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, Plots, DataFrames
    pyplot()

	"""
    `input(prompt::AbstractString="")`
  
    Read a string from STDIN. The trailing newline is stripped.
  
    The prompt string, if given, is printed to standard output without a
    trailing newline before reading input.
    """
    function input(prompt::AbstractString="")
        print(prompt)
        return chomp(readline())
    end

    export maximize_like_grad, runAll, makeData



	# methods/functions
	# -----------------

	# data creator
	# should/could return a dict with beta,numobs,X,y,norm)
	# true coeff vector, number of obs, data matrix X, response vector y, and a type of parametric distribution for G.
	function makeData(n=10000::Int, beta = [ 1; 1.5; -0.5 ]::Vector, k=3::Int)
        X = rand(MvNormal(eye(k)),n) # define X
        y = Array{Int}(n) #empty y
        for i in 1:n # define binomial y
            y[i] = rand(Bernoulli(cdf(Normal(),dot(X[:,i]',beta))))
        end
        # return a dict with beta,numobs,X,y,norm)
        return Dict("beta" => beta, "numobs" => n, "X" => X, "y" => y, "dist" => Normal())
    end


	# log likelihood function at x
	# function loglik(betas::Vector,X::Matrix,y::Vector,distrib::UnivariateDistribution) 
	function loglik(beta, d::Dict)
        l = 0
        for i in 1:d["numobs"]
            if d["y"][i] == 1
                l = l + log(cdf(d["dist"],dot(d["X"][:,i]',beta)))
            elseif d["y"][i] == 0
                l = l + log(1-cdf(d["dist"],dot(d["X"][:,i]',beta)))
            else
                println("y[$i] was not a zero or a one.")
                break
            end
        end
        return l
    end

	# gradient of the likelihood at x
	function grad!(beta::Vector, storage::Vector, d::Dict)
        g = zeros(length(beta))
        for i in 1:d["numobs"]
            if d["y"][i] == 1
                g = g + (pdf(d["dist"],dot(d["X"][:,i]',beta)))/(cdf(d["dist"],dot(d["X"][:,i]',beta)))*d["X"][:,i]
            else#if d["y"][i] == 0
                g = g - (pdf(d["dist"],dot(d["X"][:,i]',beta)))/(1-cdf(d["dist"],dot(d["X"][:,i]',beta)))*d["X"][:,i]
            #else
             #   println("y[$i] was not a zero or a one.")
              #  break
            end
        end
        storage[:] = -g
        #return g
    end

	# hessian of the likelihood at x
	function hess!(beta::Vector{Float64},storage::Matrix{Float64}, d)
        h = zeros(size(storage)) 
        #Compute the Hessian 
        for i = 1:d["numobs"]
            XB = dot(d["X"][:,i]', beta) 
            phi = pdf(d["dist"], XB) 
            Phi = cdf(d["dist"], XB) 
            XXt = d["X"][:,i]*d["X"][:,i]'
            if d["y"][i] == 1
                h = h + phi * XXt * (phi + XB * Phi) / (Phi * Phi)
            else#if d["y"][i] == 0
                h = h + (phi - XB * (1 - Phi)) / ((1 - Phi) * (1 - Phi))
            #else
             #  println("y[$i] was not a zero or a one.")
              # break
            end
        end
        storage[:] = -h 
    end


	"""
	inverse of observed information matrix
	"""
	function inv_observedInfo(betas::Vector,d)
	end

	"""
	standard errors
	"""
	function se(betas::Vector,d::Dict)
	end

	# function that maximizes the log likelihood without the gradient
	# with a call to `optimize` and returns the result
	function maximize_like(x0=[0.8,1.0,-0.1],meth=NelderMead())
        d = makeData()
        l(beta) = -loglik(beta, d)
        res = optimize(l,x0,meth)
        return res
    end


	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=BFGS())
        d = makeData()
        l(beta) = -(loglik(beta, d))
        g!(beta,storage) = grad!(beta,storage,d)
        res = optimize(l,g!,x0,meth)
        return res
    end

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=Newton())
        d = makeData()
        l(beta) = -(loglik(beta, d))
        g!(beta,storage) = grad!(beta,storage,d)
        h!(beta,storage) = hess!(beta,storage,d)
        res = optimize(l,g!,h!,x0,meth)
        return res
    end

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=BFGS())
        println("Didn't manage to get to this!")
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the the true value.
	function plotLike()
        d = makeData()
        l1(x) = loglik([ x, d["beta"][2], d["beta"][3] ], d)
        l2(x) = loglik([ d["beta"][1], x, d["beta"][3] ], d)
        l3(x) = loglik([ d["beta"][1], d["beta"][2], x ], d)
        #x1 = linspace(0, 1, 100)
        plot1 = plot(linspace(0,2,100), l1, labels = "Varying beta[1]")
        plot2 = plot(linspace(.5,2.5,100), l2, labels = "Varying beta[2]")
        plot3 = plot(linspace(-1,0,100), l3, labels = "Varying beta[3]")
        return plot(plot1, plot2, plot3)
    end


	function plotGrad()
	end


	function runAll()

        srand(1234)
		plotLike()
        l1 = loglik([1,1.5,.5], makeData())
        g1 = grad!([1,1.5,.5],zeros(3),makeData())
		m1 = maximize_like()
		m2 = maximize_like_grad()
		m3 = maximize_like_grad_hess()
		m4 = maximize_like_grad_se()
		println("results are:")
		println("maximize_like: $(m1.minimum)")
		println("maximize_like_grad: $(m2.minimum)")
		println("maximize_like_grad_hess: $(m3.minimum)")
		println("maximize_like_grad_se: $m4")
		println("")
		println("running tests:")
		include("test/runtests.jl")
		println("")
		ok = input("enter y to close this session.")
		if ok == "y"
			quit()
		end
	end


end






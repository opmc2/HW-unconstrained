

module HW_unconstrained

	# imports: which packages are we going to use in this module?
	using Distributions, Optim, PyPlot, DataFrames, Debug

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
	function loglik(d::Dict)
        l = 0
        for i in 1:d["numobs"]
            if d["y"][i] == 1
                l = l + log(cdf(d["dist"],dot(d["X"][:,i]',d["beta"])))
            else
                l = l + log(1-cdf(d["dist"],dot(d["X"][:,i]',d["beta"])))
            end
        end
        return l
    end

	# gradient of the likelihood at x
	function grad!(betas::Vector,storage::Vector,d)
	end

	# hessian of the likelihood at x
	function hessian!(betas::Vector,storage::Matrix,d)
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
	function maximize_like(x0=[0.8,1.0,-0.1],meth=:nelder_mead)
	end



	# function that maximizes the log likelihood with the gradient
	# with a call to `optimize` and returns the result
	function maximize_like_grad(x0=[0.8,1.0,-0.1],meth=:bfgs)
	end

	function maximize_like_grad_hess(x0=[0.8,1.0,-0.1],meth=:newton)
	end

	function maximize_like_grad_se(x0=[0.8,1.0,-0.1],meth=:bfgs)
	end


	# visual diagnostics
	# ------------------

	# function that plots the likelihood
	# we are looking for a figure with 3 subplots, where each subplot
	# varies one of the parameters, holding the others fixed at the true value
	# we want to see whether there is a global minimum of the likelihood at the the true value.
	function plotLike()
	end
	function plotGrad()
	end


	function runAll()

		plotLike()
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






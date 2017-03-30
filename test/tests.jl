
srand(1234)



@testset "basics" begin

	@testset "Test Data Construction" begin
	end

	@testset "Test Return value of likelihood" begin
 
	end

	@testset "Test return value of gradient" begin
        #@test length(g1) == 3
	end

	@testset "gradient vs finite difference" begin
		# gradient should not return anything,
		# but modify a vector in place.

	end
end

@testset "test maximization results" begin

	@testset "maximize returns approximate result" begin
        #@test abs(m1.minimum + l1) < 10 
	end

	@testset "maximize_grad returns accurate result" begin
        #@test abs(m2 + l1) < .1
	end

	@testset "gradient is close to zero at max like estimate" begin
        #@test abs(sum(g1)) < 1 
	end

end

@testset "test against GLM" begin

	@testset "estimates vs GLM" begin


	end

	@testset "standard errors vs GLM" begin


	end

end


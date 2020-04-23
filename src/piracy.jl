function __init__()
    @eval SparseArrays import LinearAlgebra: sym_uplo
    for op ∈ (:+, :-), Wrapper ∈ (:Hermitian, :Symmetric)
        @eval SparseArrays begin
            $op(A::AbstractSparseMatrix, B::$Wrapper{<:Any,<:AbstractSparseMatrix}) = $op(A, sparse(B))
            $op(A::$Wrapper{<:Any,<:AbstractSparseMatrix}, B::AbstractSparseMatrix) = $op(sparse(A), B)

            $op(A::AbstractSparseMatrix, B::$Wrapper) = $op(A, collect(B))
            $op(A::$Wrapper, B::AbstractSparseMatrix) = $op(collect(A), B)
        end
    end
    for op ∈ (:+, :-)
        @eval SparseArrays begin
            $op(A::Symmetric{<:Any,  <:AbstractSparseMatrix},
                B::Hermitian{<:Any,  <:AbstractSparseMatrix}) = $op(sparse(A), sparse(B))
            
            $op(A::Hermitian{<:Any,  <:AbstractSparseMatrix},
                B::Symmetric{<:Any,  <:AbstractSparseMatrix}) = $op(sparse(A), sparse(B))
            
            $op(A::Symmetric{<:Real, <:AbstractSparseMatrix},
                B::Hermitian{<:Any,  <:AbstractSparseMatrix}) = $op(Hermitian(parent(A), sym_uplo(A.uplo)), B)
            
            $op(A::Hermitian{<:Any,  <:AbstractSparseMatrix},
                B::Symmetric{<:Real, <:AbstractSparseMatrix}) = $op(A, Hermitian(parent(B), sym_uplo(B.uplo)))
        end
    end

    for f ∈ (:+, :-), (Wrapper, conjugation) ∈ ((:Hermitian, :adjoint), (:Symmetric, :transpose))
        @eval LinearAlgebra begin
            function $f(A::$Wrapper, B::$Wrapper)
                if A.uplo == B.uplo
                    return $Wrapper($f(parent(A), parent(B)), sym_uplo(A.uplo))
                elseif A.uplo == 'U'
                    return $Wrapper($f(parent(A), $conjugation(parent(B))), :U)
                else
                    return $Wrapper($f($conjugation(parent(A)), parent(B)), :U)
                end
            end
        end
    end

    for f in (:+, :-)
        @eval LinearAlgebra begin
            $f(A::Hermitian, B::Symmetric{<:Real}) = $f(A, Hermitian(parent(B), sym_uplo(B.uplo)))
            $f(A::Symmetric{<:Real}, B::Hermitian) = $f(Hermitian(parent(A), sym_uplo(A.uplo)), B)
        end
    end
end

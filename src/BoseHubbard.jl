module BoseHubbard

using LinearAlgebra, Reexport
@reexport using SparseArrays, Arpack, LowRankApprox, Conjugates, BandedMatrices, LazyBandedMatrices

export ⊗, ∑, up, dn, Bose_ladder, Hubbard_Hamiltonian, number_operator, Projector, number_projector
export Hubbard_Hamiltonian_projected

⊗(args...) = kron(args...) #Kronecker product

const ∑ = sum  #fancy sum notation

struct Up   end
struct Dn end
const up = Up()
const dn = Dn()


function Bose_ladder(;L, Ncut)
    id  = I(Ncut) |> BandedMatrix # Spinless local identity operator
    id² = id ⊗ id                # Spinful local identity operator 
    a_local = BandedMatrix(-1 => sqrt.(Ncut-1:-1:1)) #Spinless lowering operator

    a_local_up = a_local ⊗ id       # Spin up lowering operator
    a_local_dn = id      ⊗ a_local  # Spin down lowering operator
    a(i, ::Up) = reduce(⊗, (i==j ? a_local_up : id² for j ∈ 1:L)) #up lowering operator at site i
    a(i, ::Dn) = reduce(⊗, (i==j ? a_local_dn : id² for j ∈ 1:L)) #dn lowering operator at site i
end

function Hubbard_Hamiltonian(;U, μ, L, Ncut)
    a = Bose_ladder(;L=L, Ncut=Ncut)
    n(i) = a(i, up)'a(i, up) + a(i, dn)'a(i, dn) #local number operator
    Hint(i) = sparse((U*n(i) - μ*I)*n(i))        #Interation term
    
    (1/L) * (Hermitian ∘ ∑)(1:L) do i  #sum over all sites
        j = mod(i+1, 1:L)              # periodic BCs
        Ht = -1/2*((sparse(a(i, up)'a(j, up)) + hc) 
                   + (sparse(a(i, dn)'a(j, dn)) + hc)) #Kinetic term
        Ht + Hint(i)
    end
end

# Number operator for all sites
function number_operator(;L, Ncut)
    a = Bose_ladder(;L=L, Ncut=Ncut)
    n(i) = (a(i, up)'a(i, up) + a(i, dn)'a(i, dn))
    ∑(n, 1:L)
end

#Projector onto a fixed particle number
function number_projector(;L, Ncut)
    n̂ = number_operator(;L=L, Ncut=Ncut)
    function P(n)
        v = (x -> x ≈ n ? 1 : 0).(vec(n̂.data))
        sum_v = sum(v)
        inds = sortperm(v, rev=true)[1:sum_v]
        Projector(inds)
    end
end


# function Hubbard_Hamiltonian_projected(;U, μ, L, Ncut, n)
#     a = Bose_ladder(;L=L, Ncut=Ncut)
#     P = number_projector(;L=L, Ncut=Ncut)(n)
#     n̂(i) = P'*((a(i, up)'a(i, up)) + (a(i, dn)'a(i, dn)))*P
#     Hint(i) = (U*n̂(i) - μ*I)*n̂(i)
#     function Hkin(i, j)
#         aiupajup = (P'*(sparse(a(i, up)'a(j, up))))*P
#         aidnajdn = (P'*(sparse(a(i, dn)'a(j, dn))))*P
#         -((aiupajup + hc) + (aidnajdn + hc))
#     end
    
#     ∑(1:L) do i
#         j = mod(i+1, 1:L) # periodic BCs
#         Hkin(i, j) + Hint(i)
#     end * inv(L) |> Hermitian
# end

struct Projector 
    inds::Vector{Int} 
end
struct AdjointProjector
    inds::Vector{Int} 
end
struct AdjointProjectorTimesMatrix{T} 
    inds::Vector{Int}
    mat::T 
end
Base.adjoint(P::Projector) = AdjointProjector(P.inds)

Base.:(*)(Pdg::AdjointProjector, m) = AdjointProjectorTimesMatrix(Pdg.inds, m)
function Base.:(*)(PdgM::AdjointProjectorTimesMatrix, P::Projector) 
    M = PdgM.mat
    @inbounds [M[i, j] for i ∈ PdgM.inds, j ∈ P.inds]
end

function Base.:(*)(PdgM::AdjointProjectorTimesMatrix{<:BandedMatrix}, P::Projector) 
    M = PdgM.mat
    @assert M.l == M.u == 0
    @inbounds Diagonal([M[i, i] for i ∈ PdgM.inds])
end


begin # piracy 
    Base.:(+)(A::Hermitian, B::SparseMatrixCSC) = parent(A) + B
    Base.:(+)(A::SparseMatrixCSC, B::Hermitian) = A + parent(B)
    Base.:(+)(A::Hermitian{SparseMatrixCSC}, B::Hermitian{SparseMatrixCSC}) = Hermitian(parent(A) + parent(B))
end



end # module

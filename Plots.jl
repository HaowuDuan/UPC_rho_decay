cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.add("JLD2")
using DelimitedFiles
# using StaticArray
using LinearAlgebra
using LaTeXStrings
using Random
using Distributions
using FFTW
#using Plots
using Statistics
using JLD2
using FileIO
using ProgressMeter
using BenchmarkTools
#using PlotThemes
using StatsPlots
using Measures
using SpecialFunctions
using ArbNumerics
using QuadGK
#using DifferentialEquation
using ForwardDiff
using Interpolations
# This file is a test of calculation of the dipole correlator in MV model.
# All calculation will be done at fixed parameters.
#Once the code is working, it will be modifined and put into a bigger CGC package where the functions can be called.
#pre-defined functions
#SU(3) generators, t[a, i, j] is 8*3*3 matrix, T[a,b,c] is 8*8*8 matrix.
include("SU3.jl")


#longitudinal layers
const Ny=50
# Transverse lattice size in one direction
const N=64*4
#lattice size
const L=32
# lattice spacing
const a=L/N
#
const gμ=1
# infra regulator m
const m2=(0.1)^2#1/L

const Nc=3

const Ng=Nc^2-1

const αₛ=1

begin
    # define the correct wave number
    wave_number=fftfreq(N,2π)

    # Calculate magnitude of lattice momentum square for later use
    K2=zeros(N,N)
    for i in 1:N, j in 1:N
        K2[i,j]=2(2-cos(wave_number[i])-cos(wave_number[j]))/a^2
    end

    #calculate fft and ifft plans for later use
    rho=randn(ComplexF32, (N,N))
    fft_p=plan_fft(rho; flags=FFTW.MEASURE, timelimit=Inf)
    ifft_p=plan_ifft(rho; flags=FFTW.MEASURE, timelimit=Inf)

    seed=abs(rand(Int))
    rng=MersenneTwister(seed)
end




#functions to compute fundamental wilson line for a single configuration
#function to compute rho_k
function rho_k()
    # set the seed for testing purpose
    # rng = MersenneTwister()
    fft_p*rand(rng,Normal(0,gμ*1.5/(sqrt(Ny)*a)),N,N)# draw the color charge density from N(0,1)                                # for each layer, each point, and each color
end

#function to compute field A(x) for a fixed color
function Field(ρ)

        Threads.@threads for l in 1:N
            for n in 1:N
            ρ[n,l]=ρ[n,l]/(K2[n,l].+m2)
            end
        end
        # This is the problem !!!!!!!!, okie, I know why. Things blows up when m^2 is too small.
        ρ[1,1] = 0.0im

      real.(ifft_p*ρ)
end

#function to compute a single layer wilson line
function Wilson_line()
     Vₓ_arg=zeros(ComplexF32, (N,N,Nc,Nc))
     Vₓ=zeros(ComplexF32, (N,N,Nc,Nc))
     for a in 1:Ng
         ρ=rho_k()
         # This is the problem 2
         A_tmp=Field(ρ)
         for l in 1:N, n in 1:N
               Vₓ_arg[n,l,:,:]=A_tmp[n,l]*t[a]+Vₓ_arg[n,l,:,:]
         end
     end


     for l in 1:N, n in 1:N
         V_tmp=@view Vₓ_arg[n,l,:,:]
         Vₓ[n,l,:,:].= exp(1.0im*V_tmp)
     end
      Vₓ
end

function Wilson_line_Ny()
    # p=Progress(Ny)
     V=Wilson_line()
     Threads.@threads for ny in 1:Ny-1
              W=Wilson_line()
           for l in 1:N, n in 1:N
              V[n,l,:,:]=W[n,l,:,:]* V[n,l,:,:]
            end
            #next!(p)
        end
      V
end

# momentum space Wilson_line
function V_k(V)
    Vk=zeros(ComplexF32,N,N,Nc,Nc)
    for b in 1:Nc, a in 1:Nc
        Vk[:,:,a,b]=fft_p*@view V[:,:,a,b]
    end
    return Vk
end

#Momentum space dipole
function Dk(V)
       Vk=V_k(V)
       D_k=zeros(Float32,N,N)

       for j in 1:N, i in 1:N
           D_k[i,j]=@views real(tr(Vk[i,j,:,:]*adjoint(Vk[i,j,:,:])))/Nc
       end
   return D_k
end

function Dr_prime(V,Δ)
     N_step=Int(N/Δ)
     Δ_size=Δ*a

     D_k= Dk(V)
     #D_x= zeros(Float32,N,N)

     D_x= ifft(D_k)/N^2

     D_r= zeros(Float32,N_step)
     N_r= zeros(Float32,N_step)



     for i in 1: Int(N/2), j in 1:Int(N/2)
             r_index=floor(Int,sqrt(((i-1)*a)^2+((j-1)*a)^2)/Δ_size)+1
         if  r_index< N_step
             D_r[r_index]= D_r[r_index]+ real(D_x[i,j])
             N_r[r_index]= N_r[r_index]+1
         end
     end

     for r in 1:Int(N_step)
         D_tmp=D_r[r]
         D_r[r]=D_tmp/(N_r[r]+1e-6)
     end

     R=collect(1:N_step)*Δ_size
     pushfirst!(R,0)
     pushfirst!(D_r,1)

     return  (R,D_r)
end



function K(i,j)
    [(2/a)* sin(wave_number[i]/2), (2/a)* sin(wave_number[j]/2)]/(K2[i,j]+m2)
end

function ξ_generator()
     ξ_tmp=rand(rng,Normal(0,1),N,N,2,Ng)
end

function ξ_left(ξ_data, V_input)
     ξ_tmp=zeros(ComplexF32,N,N,2,Nc,Nc)
     Threads.@threads for i in 1:2
         for n in 1:N
              for m in 1:N
                   V_tmp=@view V_input[m,n,:,:]
                   V_dagger=adjoint(V_tmp)
                   ξ_tmp[m,n,i,:,:]=sum(ξ_data[m,n,i,a]*(V_tmp*t[a]*V_dagger) for a in 1:Ng)
              end

          end
      end
     return  ξ_tmp
end

function ξ_right(ξ_data,V_input)
    ξ_tmp=zeros(ComplexF32,N,N,2,Nc,Nc)
    Threads.@threads for i in 1:2
        for n in 1:N
             for m in 1:N
                  ξ_tmp[m,n,i,:,:].=sum(ξ_data[m,n,i,a]*t[a] for a in 1:Ng)
             end

         end
     end
    return ξ_tmp
end

function convolution(ξ_local)
    ξ_tmp_k=similar(ξ_local)
    ξ_tmp_k_prime=zeros(ComplexF32,N,N,Nc,Nc)
    ξ_tmp_x=zeros(ComplexF32,N,N,Nc,Nc)
    Threads.@threads for i in 1:2
        for j in 1:Nc
                for k in 1:Nc
                   ξ_tmp_k[:,:,i,j,k]=fft_p*ξ_local[:,:,i,j,k]
                end
        end
    end

    Threads.@threads for n in 1:N
        for m in 1:N
            k_tmp=K(m,n)
            ξ_tmp_k_prime[m,n,:,:]=sum(ξ_tmp_k[m,n,i,:,:]*k_tmp[i] for i in 1:2)
        end
    end

    Threads.@threads for j in 1:Nc
            for k in 1:Nc
               ξ_tmp_x[:,:,j,k]=ifft_p*ξ_tmp_k_prime[:,:,j,k]
            end
    end


    return  ξ_tmp_x
end

function exp_left(ξ_data, V_input, step)
     ξ_local=ξ_left(ξ_data, V_input)
     ξ_x=zeros(ComplexF32,N,N,Nc,Nc)
     e_left=similar(ξ_x)

     ξ_x=convolution(ξ_local)

     Threads.@threads for m in 1:N
         for n in 1:N
             e_left[n,m,:,:]=exp(2sqrt(step*αₛ)*ξ_x[n,m,:,:])
         end
     end

     return e_left
end


function exp_right(ξ_data, V_input, step)
    ξ_local=ξ_right(ξ_data, V_input)
    ξ_x=zeros(ComplexF32,N,N,Nc,Nc)
    e_right=similar(ξ_x)

    ξ_x=convolution(ξ_local)

    Threads.@threads for m in 1:N
        for n in 1:N
            e_right[n,m,:,:]=exp(-2sqrt(step*αₛ)*ξ_x[n,m,:,:])
        end
    end
    return e_right
end

function JIMWLK(Δ, N_of_steps, V_input)
      Y_f=N_of_steps*Δ

      V_f=zeros(ComplexF32,N,N,Nc,Nc)

      ξ_tmp=ξ_generator()

      factor_left=exp_left(ξ_tmp/a, V_input, 0.1)
      factor_right=exp_right(ξ_tmp/a, V_input, 0.1)

      for n in 1:N
          for m in 1:N
               V_tmp=@view V_input[m,n,:,:]
               left_tmp=@view factor_left[m,n,:,:]
               right_tmp=@view factor_right[m,n,:,:]
               W_tmp= left_tmp*V_tmp*right_tmp
               V_f[m,n,:,:].=W_tmp
          end
      end

      Threads.@threads for i in 1:N_of_steps-1
          ξ=ξ_generator()

          factor_left=exp_left(ξ, V_f, Δ)
          factor_right=exp_right(ξ, V_f, Δ)

          for n in 1:N
              for m in 1:N
                  V_tmp=@view V_f[m,n,:,:]
                  left_tmp=@view factor_left[m,n,:,:]
                  right_tmp=@view factor_right[m,n,:,:]
                  W_tmp= left_tmp*V_tmp*right_tmp
                  V_f[m,n,:,:].=W_tmp
              end
          end
      end

     V_f
end

function Sat_Mom(r,D)
          Qs=zeros(1)
          Rs=zeros(1)
          for i in 1:length(r)
                  δ=D[i]-exp(-0.5)
                 if δ<=0
                      Qs = sqrt(2)/r[i]
                      Rs = r[i]
                     break
                 end
          end
    return (Qs,Rs)
end

function Log(x)
    if x*100 >0
       return  log(x)
    else
       return 0
   end
end

function compelete_MV(x,m)
    (1-m*x*besselk(1,m*x))/m^2
end

function D_MV(r,m,Q)
    return exp(-Q^2*r^2*log(1/(r^2*(m^2)+1e-6)+exp(1))/4)
end

function D_GBW(r)
    return exp(-r^2/4)
end

function D_full_MV(r,m,Q)
    return exp(-Q^2*(1-m*r*besselk(1,m*r))/m^2)#exp(-compelete_MV(r,m))
end

function adj_dipole(f_dipole)
    return exp(log(f_dipole)*2Nc^2/Ng)
end

function WW_exact_asy(r,m)
     f_dipole=D_MV(r,0.2,0.4)
     ad= adj_dipole(f_dipole)
     Γ₁= log(1/(r^2*(m^2)+1e-6)+exp(1))-1
     Γ = r^2*log(1/(r^2*(m^2)+1e-6)+exp(1))
     xG=(1-ad)*(Γ₁-1)/Γ
     xh=(1-ad)/Γ
     return (xG,xh)
end

function WW_exact_full(r,m)
    f_dipole=D_full_MV(r,0.2,0.4)
    ad=adj_dipole(f_dipole)
    Γ=(1-m*r*besselk(1,m*r))/m^2
    Γ₁= besselk(0,m*r)/2
    Γ₂= -besselk(1,m*r)*m/(4r)
    xG=(1-ad)*(Γ₁+r^2*Γ₂)/Γ
    xh=-(1-ad)* r^2*Γ₂/Γ
    return (xG,xh)
end



function color_neutral_gamma(r,Q)
         quadgk(x ->2*(r^2)*x^3*(1-besselj(0,x))/(x^2+Q^2*r^2)/(x^2)^2 ,0,313.374,rtol=1e-9)
end

function dipole_color_neutral(r,Q,Qs)
         return exp(-Qs^2*color_neutral_gamma(r,Q)[1])
end

function Gamma_driv(r,Q)
    Gamma_cn_dat=similar(R)
    Gamma1_cn_dat=similar(R)
    Gamma2_cn_dat=similar(R)

    for i in 1:length(R)
        Gamma_cn_dat[i]=color_neutral_gamma(R[i],Q)[1]
    end

    Gamma_cn_r2= interpolate(R2, Gamma_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma1_cn_dat[i]=ForwardDiff.derivative(Gamma_cn_r2, R2[i])
    end

    Gamma1_cn_r2= interpolate(R2, Gamma1_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma2_cn_dat[i]=ForwardDiff.derivative(Gamma1_cn_r2, R2[i])
    end

    Gamma2_cn_r2= interpolate(R2, Gamma2_cn_dat, SteffenMonotonicInterpolation())

    return (Gamma1_cn_r2(r^2), Gamma2_cn_r2(r^2))
end

function WW_cn(r,Q,Qs)
    f_dipole=dipole_color_neutral(r,Q,Qs)
    ad=adj_dipole(f_dipole)
    Γ=color_neutral_gamma(r,Q)[1]

    Γ_dri=Gamma_driv(r,Q)

    Γ₁= Γ_dri[1]
    Γ₂= Γ_dri[2]

    xG= (1-ad)*(Γ₁+r^2*Γ₂)/Γ
    xh=-(1-ad)* r^2*Γ₂/Γ
    return (xG,xh,f_dipole)
end


R=LinRange(0,16,257)

d_MV=similar(R)
d_full_MV=similar(R)
d_cn=similar(R)

for i in 1:257
   # d_MV[i]=D_MV(R[i],0.2,0.48)
   # d_full_MV[i]=D_full_MV(R[i],0.2,0.4)
    d_cn[i]=dipole_color_neutral(R[i],0.5,0.454)
end

Sat_Mom(R,d_MV)[1]
Sat_Mom(R,d_full_MV)[1]
Sat_Mom(R,d_cn)[1]


plot(R*Sat_Mom(R,d_MV)[1],d_MV,label="MV small m asymptotic, m=0.2, Q_s=0.71")
plot!(R*Sat_Mom(R,d_full_MV)[1],d_full_MV,label="MV Full, m=0.2,Q_s=0.71")
plot(R*Sat_Mom(R,d_cn)[1],d_cn,label="Color neutralization scale Q=0.5, Q_s=0.71")
plot!(ylabel = L"d(r)",
            xlabel = L"rQ_s",
            box = :on,
            foreground_color_legend = nothing,
            fontfamily = "Times New Roman",
            xtickfontsize = 8,
            ytickfontsize = 8,
            xguidefontsize = 15,
            yguidefontsize = 15,
            thickness_scaling=1,
            legendfontsize=10,
            legend_font_pointsize=8,
            legendtitlefontsize=8,
            markersize=3,yguidefontrotation=-90,left_margin=10mm,bottom_margin=5mm)
xlims!(0,10)

savefig("SURGE_plots/dipoles1.pdf")


plot(R*Sat_Mom(R,d_MV)[1],-Log.(d_MV)/Sat_Mom(R,d_MV)[1]^2,label="MV small m asymptotic, m=0.2, Q_s=0.71")
plot!(R*Sat_Mom(R,d_full_MV)[1],-Log.(d_full_MV)/Sat_Mom(R,d_full_MV)[1]^2,label="MV Full, m=0.2,Q_s=0.71")
plot!(R*Sat_Mom(R,d_cn)[1],-Log.(d_cn)/Sat_Mom(R,d_cn)[1]^2,label="Color neutralization scale Q=0.5,Q_s=0.71")
plot!(ylabel = L"-\frac{\ln d(r)}{Q_s^2}",
            xlabel = L"rQ_s",
            box = :on,
            foreground_color_legend = nothing,
            fontfamily = "Times New Roman",
            xtickfontsize = 8,
            ytickfontsize = 8,
            xguidefontsize = 12,
            yguidefontsize = 12,
            thickness_scaling=1,
            legendfontsize=10,
            legend_font_pointsize=8,
            legendtitlefontsize=8,
            markersize=3,yguidefontrotation=-90,left_margin=12mm,bottom_margin=5mm)
xlims!(0,10)

plot!(ylabel = L"\Gamma(r)",
    xlabel = L"rQ_s")
savefig("SURGE_plots/Gammas1.pdf")
xlims!(0,40)



# Caculate instead of dipole, xG, xh

xG_a=similar(R)
xh_a=similar(R)

xG_f=similar(R)
xh_f=similar(R)

for i in 1:2000
    asy_tmp=WW_exact_asy(R[i],0.1)
    full_tmp=WW_exact_full(R[i],0.1)
    xG_a[i]=asy_tmp[1]
    xh_a[i]=asy_tmp[2]

    xG_f[i]=full_tmp[1]
    xh_f[i]=full_tmp[2]
end

plot(R*Sat_Mom(R,d_MV)[1],xG_a,label="xG, MV small m asymptotic, Q_s=0.71")
plot!(R*Sat_Mom(R,d_full_MV)[1],xG_f,label="xG, MV full, Q_s=0.71")
plot!(ylabel = L"xG^{(1)}(r)",
        xlabel = L"rQ_s",
        box = :on,
        foreground_color_legend = nothing,
        fontfamily = "Times New Roman",
        xtickfontsize = 8,
        ytickfontsize = 8,
        xguidefontsize = 15,
        yguidefontsize = 15,
        thickness_scaling=1,
        legendfontsize=10,
        legend_font_pointsize=8,
        legendtitlefontsize=8,
        markersize=3,yguidefontrotation=-90,left_margin=12mm,bottom_margin=5mm)
xlims!(0,10)


savefig("xG_exact.pdf")

plot(R*Sat_Mom(R,d_MV)[1],xh_a,label="xh, MV small m asymptotic")
plot!(R*Sat_Mom(R,d_full_MV)[1],xh_f,label="xh, MV full")
plot!(ylabel = L"xh^{(1)}(r)",
        xlabel = L"rQ_s",
        box = :on,
        foreground_color_legend = nothing,
        fontfamily = "Times New Roman",
        xtickfontsize = 8,
        ytickfontsize = 8,
        xguidefontsize = 15,
        yguidefontsize = 15,
        thickness_scaling=1,
        legendfontsize=10,
        legend_font_pointsize=8,
        legendtitlefontsize=8,
        markersize=3,yguidefontrotation=-90,left_margin=12mm,bottom_margin=5mm)
xlims!(0,20)

savefig("xh_exact.pdf")


xG_cn=similar(R)
xh_cn=similar(R)

R2=R.^2

d_cn=similar(R)

p=Progress(2000)

Threads.@threads for i in 1:2000
    tmp= WW_cn(R[i],0.5,0.454)
    xG_cn[i]=tmp[1]
    xh_cn[i]=tmp[2]

    d_cn[i]=tmp[3]
    next!(p)
end

Sat_Mom(R,d_cn)

Ob_1=similar(R)
Ob_0=similar(R)
for i in 1:2000
    Ob_0[i]=R2[i]*xh_f[i]/(1-adj_dipole(d_full_MV[i]))
    Ob_1[i]=R2[i]*xh_cn[i]/(1-adj_dipole(d_cn[i]))
end

# Compare xG, xh


plot(R*Sat_Mom(R,d_cn)[1],xG_cn,label="xG, Q=0.5,Q_s=0.71")
plot!(R*Sat_Mom(R,d_full_MV)[1],xG_f,label="xG, MV, m=0.2,Q_s=0.71")
plot!(ylabel = L"xG^{(1)}(r)",
        xlabel = L"rQ_s",
        box = :on,
        foreground_color_legend = nothing,
        fontfamily = "Times New Roman",
        xtickfontsize = 8,
        ytickfontsize = 8,
        xguidefontsize = 15,
        yguidefontsize = 15,
        thickness_scaling=1,
        legendfontsize=10,
        legend_font_pointsize=8,
        legendtitlefontsize=8,
        markersize=3,yguidefontrotation=-90,left_margin=14mm,bottom_margin=2mm)
xlims!(0,10)

savefig("SURGE_plots/xG_compare.pdf")

plot(R*Sat_Mom(R,d_cn)[1],xh_cn,label="xh, Q=0.5, Q_s=0.71")
plot!(R*Sat_Mom(R,d_full_MV)[1],xh_f,label="xh, MV, m=0.2, Q_s=0.71")
plot!(ylabel = L"xh^{(1)}(r)",
        xlabel = L"rQ_s",
        box = :on,
        foreground_color_legend = nothing,
        fontfamily = "Times New Roman",
        xtickfontsize = 8,
        ytickfontsize = 8,
        xguidefontsize = 15,
        yguidefontsize = 15,
        thickness_scaling=1,
        legendfontsize=10,
        legend_font_pointsize=8,
        legendtitlefontsize=8,
        markersize=3,yguidefontrotation=-90,left_margin=12mm,bottom_margin=2mm)
xlims!(0,10)


savefig("SURGE_plots/xh_compare.pdf")

plot(R*Sat_Mom(R,d_cn)[1],xG_cn./(1 .-adj_dipole.(d_cn)),label="xG, Q=0.5,Q_s=0.71")
plot!(R*Sat_Mom(R,d_full_MV)[1],xG_f./(1 .-adj_dipole.(d_full_MV)),label="xG, MV, m=0.2, Q_s=0.71")
plot!(ylabel = L"\frac{xG^{(1)}(r)}{1-D(r)}",
        xlabel = L"rQ_s",
        box = :on,
        foreground_color_legend = nothing,
        fontfamily = "Times New Roman",
        xtickfontsize = 8,
        ytickfontsize = 8,
        xguidefontsize = 15,
        yguidefontsize = 15,
        thickness_scaling=1,
        legendfontsize=10,
        legend_font_pointsize=8,
        legendtitlefontsize=8,
        markersize=3,yguidefontrotation=-90,left_margin=16mm,bottom_margin=5mm)
xlims!(0,1)

savefig("SURGE_plots/xG_compare2.pdf")

plot(R*Sat_Mom(R,d_cn)[1],xh_cn./(1 .-adj_dipole.(d_cn)),label="xh, Q=0.5")
plot!(R*Sat_Mom(R,d_full_MV)[1],xh_f./(1 .-adj_dipole.(d_full_MV)),label="xh, MV, m=0.2")
plot!(ylabel = L"\frac{xh^{(1)}(r)}{1-D(r)}",
        xlabel = L"rQ_s",
        box = :on,
        foreground_color_legend = nothing,
        fontfamily = "Times New Roman",
        xtickfontsize = 8,
        ytickfontsize = 8,
        xguidefontsize = 15,
        yguidefontsize = 15,
        thickness_scaling=1,
        legendfontsize=10,
        legend_font_pointsize=8,
        legendtitlefontsize=8,
        markersize=3,yguidefontrotation=-90,left_margin=16mm,bottom_margin=5mm)
xlims!(0,1)

savefig("SURGE_plots/xh_compare2.pdf")
print(xG_f)
plot(R*Sat_Mom(R,d_cn)[1],R.^2 .*xG_cn./(1 .-adj_dipole.(d_cn)),label="xG, Q=0.5, Q_s=0.71")
plot!(R*Sat_Mom(R,d_full_MV)[1],R.^2 .*xG_f./(1 .-adj_dipole.(d_full_MV)),label="xG, MV, m=0.2, Q_s=0.71")
plot!(ylabel = L"\frac{r^2xG^{(1)}(r)}{1-D(r)}",
        xlabel = L"rQ_s",
        box = :on,
        foreground_color_legend = nothing,
        fontfamily = "Times New Roman",
        xtickfontsize = 8,
        ytickfontsize = 8,
        xguidefontsize = 15,
        yguidefontsize = 15,
        thickness_scaling=1,
        legendfontsize=10,
        legend_font_pointsize=8,
        legendtitlefontsize=8,
        markersize=3,yguidefontrotation=-90,left_margin=18mm,bottom_margin=5mm)

xlims!(0,10)

savefig("SURGE_plots/xG_compare3.pdf")
Sat_Mom(R,d_cn)[1]
plot(R*Sat_Mom(R,d_cn)[1],R.^2 .*xh_cn./(1 .-adj_dipole.(d_cn)),label="xh, Q=0.5, Q_s=0.71")
plot!(R*Sat_Mom(R,d_full_MV)[1],R.^2 .*xh_f./(1 .-adj_dipole.(d_full_MV)),label="xh, MV, m=0.2, Q_s=0.71")
plot!(ylabel = L"\frac{r^2xh^{(1)}(r)}{1-D(r)}",
        xlabel = L"rQ_s",
        box = :on,
        foreground_color_legend = nothing,
        fontfamily = "Times New Roman",
        xtickfontsize = 8,
        ytickfontsize = 8,
        xguidefontsize = 15,
        yguidefontsize = 15,
        thickness_scaling=1,
        legendfontsize=10,
        legend_font_pointsize=8,
        legendtitlefontsize=8,
        markersize=3,yguidefontrotation=-90,left_margin=18mm,bottom_margin=5mm)
xlims!(0,12)

savefig("SURGE_plots/xh_compare3.pdf")

# Vladi's data
dipole_vs=zeros(255,2,10)
xg_vs=zeros(255,4,10)
for i in 1:10
     y=i-1
     dipole_vs[:,:,i]=readdlm("vs_data/S_0.$y.dat")
end

Qs = zeros(5)

for i in 1:5
    Qs[i] = Sat_Mom(dipole_vs[:, 1, i*2], dipole_vs[:, 2, i*2])[1]
end

print(Qs)

for i in 1:2000
    d_full_MV[i] = D_full_MV(R[i], 0.2, 0.345)
    d_cn[i] = dipole_color_neutral(R[i], 0.4, 0.5)
end

Sat_Mom(R, d_MV)[1]
Sat_Mom(R, d_full_MV)[1]
Sat_Mom(R, d_cn)[1]

plot(R * Sat_Mom(R, d_MV)[1], -Log.(d_MV) / Sat_Mom(R, d_MV)[1]^2, label="MV small m asymptotic, m=0.2,",)
plot(R * Sat_Mom(R, d_full_MV)[1], -Log.(d_full_MV) / Sat_Mom(R, d_full_MV)[1]^2, label="MV Full, m=0.2, Q_s=0.561", linestyle=:dash, linecolor=:black)
plot!(R * Sat_Mom(R, d_cn)[1], -Log.(d_cn) / Sat_Mom(R, d_cn)[1]^2, label="Color neutralized, Q_s=0.9", linecolor=:black)
plot!(ylabel=L"-\frac{\ln d(r)}{Q_s^2}",
    xlabel=L"rQ_s",
    box=:on,
    foreground_color_legend=nothing,
    fontfamily="Times New Roman",
    xtickfontsize=8,
    ytickfontsize=8,
    xguidefontsize=12,
    yguidefontsize=12,
    thickness_scaling=1,
    legendfontsize=10,
    legend_font_pointsize=8,
    legendtitlefontsize=8,
    markersize=3, yguidefontrotation=-90, left_margin=12mm, bottom_margin=5mm)
xlims!(0, 10)



plot!(dipole_vs[:, 1, 1] * Qs[1], -Log.(dipole_vs[:, 2, 1]) / Qs[1]^2, label="y=0, Q_s=0.57")
plot!(dipole_vs[:, 1, 3] * Qs[2], -Log.(dipole_vs[:, 2, 3]) / Qs[2]^2, label="y=0.2")
plot!(dipole_vs[:, 1, 5] * Qs[3], -Log.(dipole_vs[:, 2, 5]) / Qs[3]^2, label="y=0.4, Q_s=0.87")
plot!(dipole_vs[:, 1, 7] * Qs[4], -Log.(dipole_vs[:, 2, 7]) / Qs[4]^2, label="y=0.6")
plot!(dipole_vs[:, 1, 9] * Qs[5], -Log.(dipole_vs[:, 2, 9]) / Qs[5]^2, label="y=0.8, Q_s=1.51")


plot!(ylabel=L"-\frac{\ln d(r)}{Q_s^2}",
    xlabel=L"rQ_s",
    box=:on,
    foreground_color_legend=nothing,
    fontfamily="Times New Roman",
    xtickfontsize=8,
    ytickfontsize=8,
    xguidefontsize=15,
    yguidefontsize=15,
    thickness_scaling=1,
    legendfontsize=10,
    legend_font_pointsize=8,
    legendtitlefontsize=8,
    markersize=3, yguidefontrotation=-90, left_margin=18mm, bottom_margin=5mm)


scatter!(dipole_vs[1:8:250, 1, 1] * Qs[1], -Log.(dipole_vs[1:8:250, 2, 1]) / Qs[1]^2, label="y=0, Q_s=0.57", markersize=3)
#plot!(dipole_vs[:, 1, 3] * Qs[2], -Log.(dipole_vs[:, 2, 3]) / Qs[2]^2, label="y=0.2")
scatter!(dipole_vs[1:8:250, 1, 5] * Qs[3], -Log.(dipole_vs[1:8:250, 2, 5]) / Qs[3]^2, label="y=0.4, Q_s=0.87",markersize=3)
#plot!(dipole_vs[:, 1, 7] * Qs[4], -Log.(dipole_vs[:, 2, 7]) / Qs[4]^2, label="y=0.6")
scatter!(dipole_vs[1:8:250, 1, 9] * Qs[5], -Log.(dipole_vs[1:8:250, 2, 9]) / Qs[5]^2, label="y=0.8, Q_s=1.51",markersize=3)

xlims!(0,10)
ylims!(0,12)
savefig("SURGE_plots/Gamma_JIMWLK.pdf")
# data for xh

vs_xh = zeros(255, 4, 10)

for i in 1:9
    vs_xh[:, :, i] = readdlm("vs_data/xh_0.$i.dat")
end

plot(vs_xh[:, 1, 1], vs_xh[:, 1, 1].^2 .*vs_xh[:, 2, 1] ./ (1 .- vs_xh[:, 3, 1]),label="y=0")

plot!(vs_xh[:, 1, 3], vs_xh[:, 1, 3] .^ 2 .* vs_xh[:, 2, 3] ./ (1 .- vs_xh[:, 3, 3]),label="y=0.2")

plot!(vs_xh[:, 1, 5], vs_xh[:, 1, 5] .^ 2 .* vs_xh[:, 2, 5] ./ (1 .- vs_xh[:, 3, 5]),label="y=0.4")

plot!(vs_xh[:, 1, 7], vs_xh[:, 1, 7] .^ 2 .* vs_xh[:, 2, 7] ./ (1 .- vs_xh[:, 3, 7]),label="y=0.6")

xlims!(0,5)

savefig("vs_xh.pdf")

#my data for xh

hwd_xh=zeros(6000,129,10)

for i in 1:6000
    data_tmp = readdlm("/Users/hwd/Desktop/JIMWLK_and_Gaussian/xgh_HPC/corr_xgh/correct_xgh_$(i).dat")
    hwd_xh[i, :, :] = reshape(data_tmp, (129, 7))
end



function color_neutral_gamma2(r,Q,γ,m)
         quadgk(x ->2*(1/Q^2)/π*x^3*(1-besselj(0,x))/(x^2+m^2*r^2)^2*(1-1/(1+(Q^2*r^2/x^2)^γ)) ,0,313.374,rtol=1e-9)
end

function dipole_color_neutral2(r,Q,Qs,γ,m)
         return exp(-Qs^2*color_neutral_gamma2(r,Q,γ,m)[1])
end

function Gamma_driv2(r,Q,γ,m)
    Gamma_cn_dat=similar(R)
    Gamma1_cn_dat=similar(R)
    Gamma2_cn_dat=similar(R)

    for i in 1:length(R)
        Gamma_cn_dat[i]=color_neutral_gamma2(R[i],Q,γ,m)[1]
    end

    Gamma_cn_r2= interpolate(R2, Gamma_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma1_cn_dat[i]=ForwardDiff.derivative(Gamma_cn_r2, R2[i])
    end

    Gamma1_cn_r2= interpolate(R2, Gamma1_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma2_cn_dat[i]=ForwardDiff.derivative(Gamma1_cn_r2, R2[i])
    end

    Gamma2_cn_r2= interpolate(R2, Gamma2_cn_dat, SteffenMonotonicInterpolation())

    return (Gamma1_cn_r2(r^2), Gamma2_cn_r2(r^2))
end

function Gamma_driv3(Gamma_cn_dat)
    Gamma1_cn_dat=similar(R)
    Gamma2_cn_dat=similar(R)

    #for i in 1:length(R)
    #    Gamma_cn_dat[i]=color_neutral_gamma2(R[i],Q,γ,m)[1]
    #end

    Gamma_cn_r2= interpolate(R2,Gamma_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma1_cn_dat[i]=ForwardDiff.derivative(Gamma_cn_r2, R2[i])
    end

    Gamma1_cn_r2= interpolate(R2, Gamma1_cn_dat, SteffenMonotonicInterpolation())

    for i in 1:length(R)
        Gamma2_cn_dat[i]=ForwardDiff.derivative(Gamma1_cn_r2, R2[i])
    end

    #Gamma2_cn_r2= interpolate(R2, Gamma2_cn_dat, SteffenMonotonicInterpolation())

    return [Gamma1_cn_dat,Gamma2_cn_dat]
end

function WW_cn3(Gamma_cn_dat,adj_dat)
    #f_dipole=dipole_color_neutral2(r,Q,Qs,γ,m)
    ad=adj_dat
    Γ=Gamma_cn_dat

    xG=similar(Gamma_cn_dat)
    xh=similar(Gamma_cn_dat)
    r1=similar(xG)
    r2=similar(xh)

    Γ_dri=Gamma_driv3(Γ)

    Γ₁= Γ_dri[1]
    Γ₂= Γ_dri[2]

    xG .=#= (1 .-ad) .*=# (Γ₁.+ R .^2 .* Γ₂) #./Γ
    xh .=#=-(1 .-ad) .*=# -R .^2 .*Γ₂ #./Γ

    r1.= xG ./xh
    r2.= (xG .- xh)./ (xG .+ xh)

    return [r1,r2]
end



function WW_cn2(r,Q,Qs,γ,m)
    f_dipole=dipole_color_neutral2(r,Q,Qs,γ,m)
    ad=adj_dipole(f_dipole)

    Γ=color_neutral_gamma2(r,Q,γ,m)[1]

    Γ_dri=Gamma_driv2(r,Q,γ,m)

    Γ₁= Γ_dri[1]
    Γ₂= Γ_dri[2]

    xG= (1-ad)*(Γ₁+r^2*Γ₂)/Γ
    xh=-(1-ad)* r^2*Γ₂/Γ
    return (xG,xh,f_dipole)
end

R=LinRange(0,20,2000)

R2=R .^2

data_ratio=zeros(2000,2,5,2)
Q=[0.1,0.5,1,2,4]
p=Progress(2000)

for j in 1:5

         tmp_dipole=similar(R)
         tmp_adj_dipole=similar(R)
         tmp_gamma=similar(R)

         Threads.@threads for i in 1:2000
             tmp_dipole[i]=dipole_color_neutral2(R[i],Q[j],0.454,0.7,0.2)
             tmp_gamma[i]=color_neutral_gamma2(R[i],Q[j],0.7,0.2)[1]
         end


         tmp_adj_dipole=adj_dipole.(tmp_dipole)

         tmp= WW_cn3(tmp_gamma,tmp_adj_dipole)

         data_ratio[:,1,j,1].=tmp[1]
         data_ratio[:,2,j,1].=tmp[2]

         #data_xgh[i,3,j,1]=tmp[3]
    end
end

data_ratio[:,1,1,1]

plot(R[1:1000], data_ratio[1:1000,1,1,1] , label="m=0.2, Q=0.1")
plot!(R[1:1000], data_ratio[1:1000,1,2,1] , label="m=0.2, Q=0.5")
plot!(R[1:1000], data_ratio[1:1000,1,3,1] , label="m=0.2, Q=1")
plot!(R[1:1000], data_ratio[1:1000,1,4,1] , label="m=0.2, Q=2")
plot!(R[1:1000], data_ratio[1:1000,1,5,1] , label="m=0.2, Q=4")
xlabel!("r")
ylabel!(L"\frac{xg}{xh}")
plot!(yguidefontrotation=-90,leftmargin=15mm)

plot(R[1:1000], data_ratio[1:1000,2,1,1] , label="m=0.2, Q=0.1")
plot!(R[1:1000],data_ratio[1:1000,2,2,1] , label="m=0.2, Q=0.5")
plot!(R[1:1000],data_ratio[1:1000,2,3,1] , label="m=0.2, Q=1")
plot!(R[1:1000],data_ratio[1:1000,2,4,1] , label="m=0.2, Q=2")
plot!(R[1:1000],data_ratio[1:1000,2,5,1] , label="m=0.2, Q=4")
xlabel!("r")
ylabel!(L"\frac{xg-xh}{xg+xh}")
plot!(yguidefontrotation=-90,leftmargin=15mm,xlims=(0,5),ylims=(-2.5,1))

plot(R[1:1000], data_ratio[1:1000,1,1,2] , label="m=0.001, Q=0.1")
plot!(R[1:1000], data_ratio[1:1000,1,2,2] , label="m=0.001, Q=0.5")
plot!(R[1:1000], data_ratio[1:1000,1,3,2] , label="m=0.001, Q=1")
plot!(R[1:1000], data_ratio[1:1000,1,4,2] , label="m=0.001, Q=2")
plot!(R[1:1000], data_ratio[1:1000,1,5,2] , label="m=0.001, Q=4")
xlabel!("r")
ylabel!(L"\frac{xg}{xh}")
plot!(yguidefontrotation=-90,leftmargin=15mm)

plot(R[1:1000], data_ratio[1:1000,2,1,2] , label="m=0.001, Q=0.1")
plot!(R[1:1000],data_ratio[1:1000,2,2,2] , label="m=0.001, Q=0.5")
plot!(R[1:1000],data_ratio[1:1000,2,3,2] , label="m=0.001, Q=1")
plot!(R[1:1000],data_ratio[1:1000,2,4,2] , label="m=0.001, Q=2")
plot!(R[1:1000],data_ratio[1:1000,2,5,2] , label="m=0.001, Q=4")
xlabel!("r")
ylabel!(L"\frac{xg-xh}{xg+xh}")
plot!(yguidefontrotation=-90,leftmargin=15mm)




function check(Gamma_cn_dat,adj_dat)
    #f_dipole=dipole_color_neutral2(r,Q,Qs,γ,m)
    ad=adj_dat
    Γ=Gamma_cn_dat

    xG=similar(Gamma_cn_dat)
    xh=similar(Gamma_cn_dat)
    r1=similar(xG)
    r2=similar(xh)

    Γ_dri=Gamma_driv3(Γ)

    Γ₁= Γ_dri[1]
    Γ₂= Γ_dri[2]

    xG .=#= (1 .-ad) .*=# (Γ₁.+ R .^2 .* Γ₂) #./Γ
    xh .=#=-(1 .-ad) .*=# -R .^2 .*Γ₂ #./Γ

    #r1.= xG ./xh
    #r2.= (xG .- xh)./ (xG .+ xh)

    return [xG,xh]
end

data_check=zeros(2000,2,5,2)

for j in 1:5

         tmp_dipole=similar(R)
         tmp_adj_dipole=similar(R)
         tmp_gamma=similar(R)

         Threads.@threads for i in 1:2000
             tmp_dipole[i]=dipole_color_neutral2(R[i],Q[j],0.454,0.7,0.001)
             tmp_gamma[i]=color_neutral_gamma2(R[i],Q[j],0.7,0.001)[1]
         end


         tmp_adj_dipole=adj_dipole.(tmp_dipole)

         tmp= WW_cn3(tmp_gamma,tmp_adj_dipole)

         data_check[:,1,j,2].=tmp[1]
         data_check[:,2,j,2].=tmp[2]

         #data_xgh[i,3,j,1]=tmp[3]
    end
end


plot(R[1:1000], data_check[1:1000,2,1,1] , label="m=0.2, Q=0.1 , xh")
plot(R[100:1500], R[100:1500] .* data_check[100:1500,2,1,1] , label="m=0.2, Q=0.1, r*xh")

plot(R[100:1500], R[100:1500] .* data_check[100:1500,2,1,2] , label="m=0.001, Q=0.1, r*xh")


# gamma=1 xG/(1-D(r))
Q1=LinRange(0.1,10,100)

function WW_cn4(Gamma_cn_dat)
    #f_dipole=dipole_color_neutral2(r,Q,Qs,γ,m)
    #ad=adj_dat
    Γ=Gamma_cn_dat

    xG=similar(Gamma_cn_dat)
    xh=similar(Gamma_cn_dat)
    #r1=similar(xG)
    #r2=similar(xh)

    Γ_dri=Gamma_driv3(Γ)

    Γ₁= Γ_dri[1]
    Γ₂= Γ_dri[2]

    xG .=#= (1 .-ad) .*=# (Γ₁.+ R .^2 .* Γ₂)# ./Γ
    xh .=#=-(1 .-ad) .*=# -R .^2 .*Γ₂ #./Γ

    #r1.= xG ./xh
    #r2.= (xG .- xh)./ (xG .+ xh)

    return [xG,xh]
end

function Sat_Mom(r,D)
          Qs=zeros(1)
          Rs=zeros(1)
          for i in 1:length(r)
                  δ=D[i]-exp(-0.5)
                 if δ<=0
                      Qs = sqrt(2)/r[i]
                      Rs = r[i]
                     break
                 end
          end
    return (Qs,Rs)
end
function R_max(R,data)
    r_max=zeros(1)
    for i in 1:length(R)
           δ=data[i]-data[i+1]
           if δ>=0
            #    Qs = sqrt(2)/r[i]
                r_max = data[i]
               break
           end
    end
    return r_max
end


end
data_gamma1=zeros(2000,2,100)
p1=Progress(100)

Threads.@threads for j in 1:100

            # tmp_dipole=similar(R)
            # tmp_adj_dipole=similar(R)
             tmp_gamma=similar(R)

             for i in 1:2000
                # tmp_dipole[i]=dipole_color_neutral2(R[i],Q[j],0.454,1,0.001)
                 tmp_gamma[i]=color_neutral_gamma2(R[i],Q1[j],1,0.001)[1]
             end


            # tmp_adj_dipole=adj_dipole.(tmp_dipole)

             tmp= WW_cn4(tmp_gamma)

             data_gamma1[:,1,j].=tmp[1]
             data_gamma1[:,2,j].=tmp[2]

             #data_xgh[i,3,j,1]=tmp[3]
        end
      next!(p1)
end


plot(R, R  .* data_gamma1[:,2,20], label="Q=0.2")
plot!(R, R .* data_gamma1[:,2,40],label="Q=0.4")
plot!(R, R  .* data_gamma1[:,1,60],label="Q=0.6")
xlims!(0,5)
plot!(ylabel=L"\frac{rxG}{1-D}")

R_M=similar(Q1)

for i in 1:length(R_M)
    R_M[i]=R_max(R,data_gamma1[:,1,i])
end

plot(Q1,R_M .* Q1, label=L"\gamma=1",ylabel=L"r_{max}Q",yguidefontrotation=-90,leftmargin=15mm)

data_gamma2=zeros(2000,2,100)

Threads.@threads for j in 1:100

            # tmp_dipole=similar(R)
            # tmp_adj_dipole=similar(R)
             tmp_gamma=similar(R)

             for i in 1:2000
                # tmp_dipole[i]=dipole_color_neutral2(R[i],Q[j],0.454,1,0.001)
                 tmp_gamma[i]=color_neutral_gamma2(R[i],Q1[j],0.7,0.001)[1]
             end


            # tmp_adj_dipole=adj_dipole.(tmp_dipole)

             tmp= WW_cn4(tmp_gamma)

             data_gamma2[:,1,j].=tmp[1]
             data_gamma2[:,2,j].=tmp[2]

             #data_xgh[i,3,j,1]=tmp[3]
        end

end

R_M2=similar(Q1)

for i in 1:length(R_M2)
    R_M2[i]=R_max(R,data_gamma2[:,1,i])
end

plot!(Q1,R_M2 .* Q1, label=L"\gamma=0.7",xlabel=L"Q" ,yguidefontrotation=-90,leftmargin=15mm)


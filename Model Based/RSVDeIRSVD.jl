######################################################################################################################################
# Implementação dos métodos baseados em Modelo: RSVD e IRSVD
#
# Introdução a Sistemas de Recomendação - 2016/2
# Universidade Federal Rural do Rio de Janeiro
# Integrantes: Lívia de Azevedo, Gustavo Ebbo e Ivo Paiva
#
######################################################################################################################################

#Com relação a base de dados
qtd_users = 943
qtd_items = 1682

#Lendo a base de dados do MovieLens de 100K
data = readdlm("u.data")
data = convert(Array{Int64},data)

##########################################################################################

#Montando as bases de dados que serão usadas nos testes
#Conjuntos retirados do Movie Lens.

#"The data sets ua.base e ua.test split the u data into a training set and a test set with 
#exactly 10 ratings per user in the test set."

train1 = readdlm("ua.base") #90570 elementos ----> 90,57%
test1 = readdlm("ua.test") #9430 elementos ------> 9,43%

train1 = convert(Array{Int64},train1)
test1 = convert(Array{Int64},test1)

#Divisão aleatória da base de dados em teste e treinamento

training_size = 80000
test_size = 20000

train2 = zeros(training_size,3) #80000 elementos ------> 80%
test2 = zeros(test_size,3) #20000 elementos --------> 20%

seed = convert(Int64,round(rand()*10000+1)) #Gera seed
ordem_aleatoria = shuffle(MersenneTwister(seed),collect(1:size(data)[1]))

escolhas_treinamento = find(r->r, ordem_aleatoria .> test_size)
escolhas_teste = find(r->r, !(ordem_aleatoria .> test_size))

train2[1:training_size,1:3] = data[escolhas_treinamento[1:training_size],1:3]
test2[1:test_size,1:3] = data[escolhas_teste[1:test_size],1:3]

train2 = convert(Array{Int64},train2)
test2 = convert(Array{Int64},test2)

##########################################################################################

#Funções das métricas de erro para cada modelo.

function root_squared_mean_error_RSVD(test::AbstractArray,q::AbstractArray,p::AbstractArray)
    RSME = 0.0
    for i=1:size(test)[1]
        RSME += (test[i,3] - dot(vec(q[test[i,2],:]),vec(p[test[i,1],:])))^2
    end
    RSME /= size(test)[1]
    RSME = sqrt(RSME)
    
    return RSME
end

function root_squared_mean_error_IRSVD(test::AbstractArray,bias_item::AbstractArray,bias_user::AbstractArray,
                                       mean_global_train::Float64,q::AbstractArray,p::AbstractArray)
    RSME = 0.0
    for i=1:size(test)[1]
        RSME += (test[i,3] - (mean_global_train + bias_item[test[i,2]] + bias_user[test[i,1]] + 
                dot(vec(q[test[i,2],:]),vec(p[test[i,1],:]))))^2
    end
    RSME /= size(test)[1]
    RSME = sqrt(RSME)
    
    return RSME
end

##########################################################################################

#Implementação RSVD

#Parâmetros do método(Já determinados como ótimos pelo seu criador, Simon Funk, para o Netflix Prize)
#lambda = 0.02
#learning_rate = 0.001
function Regulared_SVD(training_set::AbstractArray,test_set::AbstractArray,lambda::Float64,learning_rate::Float64,stop_criteria::Float64,q::AbstractArray,p::AbstractArray)

    error_prediction = zeros(qtd_users,qtd_items)
    current_error::Float64 = size(data)[1]
    next_error::Float64 = 0.0

    while abs(current_error - next_error) > stop_criteria

        current_error = next_error

        for i=1:size(training_set)[1]                    
            q[training_set[i,2],:] += learning_rate * ( ( training_set[i,3] - dot(vec(q[training_set[i,2],:]),
                                      vec(p[training_set[i,1],:])) ) * p[training_set[i,1],:] - 
                                      lambda * q[training_set[i,2],:])

            p[training_set[i,1],:] += learning_rate * ( ( training_set[i,3] - dot(vec(q[training_set[i,2],:]),
                                      vec(p[training_set[i,1],:])) ) * q[training_set[i,2],:] - 
                                      lambda * p[training_set[i,1],:])
        end
        
        next_error = 0

        for i=1:size(training_set)[1]
            next_error += ((training_set[i,3] - dot(vec(q[training_set[i,2],:]),
                             vec(p[training_set[i,1],:])))^2 + lambda * (norm(q[training_set[i,2],:])^2 + 
                             norm(p[training_set[i,1],:])^2)) 
        end
    end

    #Previsão do conjunto de teste
    RSME = root_squared_mean_error_RSVD(test_set,q,p)

    return RSME
end

#Implementação IRSVD

#Parâmetros do método já definidos como os melhores
#lambda = 0.02
#learning_rate = 0.005
function Improved_Regulared_SVD(training_set::AbstractArray,test_set::AbstractArray,lambda::Float64,learning_rate::Float64,stop_criteria::Float64,q::AbstractArray,p::AbstractArray)
    
    #Para inicialização, os bias começarão com o valor zero.
    bias_item = zeros(qtd_items)
    bias_user = zeros(qtd_users)

    ##########################################################################################

    current_error::Float64 = size(data)[1]
    next_error::Float64 = 0.0

    #Definindo a média global do conjunto de treinamento
    mean_global_train = mean(training_set[:,3])

    while abs(current_error - next_error) > stop_criteria

        current_error = next_error

        #Atualização de "b(i)","b(u)","q" e "p"
        for i=1:size(training_set)[1]
            bias_user[training_set[i,1]] += learning_rate * ( (training_set[i,3] - 
                                            (mean_global_train + bias_item[training_set[i,2]] + bias_user[training_set[i,1]] + 
                                            dot(vec(q[training_set[i,2],:]),vec(p[training_set[i,1],:]))) ) -
                                            lambda * bias_user[training_set[i,1]])
            bias_item[training_set[i,2]] += learning_rate * ( (training_set[i,3] - 
                                            (mean_global_train + bias_item[training_set[i,2]] + bias_user[training_set[i,1]] + 
                                            dot(vec(q[training_set[i,2],:]),vec(p[training_set[i,1],:]))) ) -
                                            lambda * bias_item[training_set[i,2]])
            q[training_set[i,2],:] += learning_rate * ( (training_set[i,3] - 
                                      (mean_global_train + bias_item[training_set[i,2]] + bias_user[training_set[i,1]] + 
                                      dot(vec(q[training_set[i,2],:]),vec(p[training_set[i,1],:]))) ) * 
                                      p[training_set[i,1],:] - lambda * q[training_set[i,2],:])
            p[training_set[i,1],:] += learning_rate * ((training_set[i,3] - 
                                      (mean_global_train + bias_item[training_set[i,2]] + bias_user[training_set[i,1]] + 
                                      dot(vec(q[training_set[i,2],:]),vec(p[training_set[i,1],:]))) ) * 
                                      q[training_set[i,2],:] - lambda * p[training_set[i,1],:])
        end

        next_error = 0

        for i=1:size(training_set)[1]
            next_error += (training_set[i,3] - mean_global_train - bias_item[training_set[i,2]] - bias_user[training_set[i,1]] - 
                          dot(vec(q[training_set[i,2],:]),vec(p[training_set[i,1],:])))^2 + 
                          lambda * (bias_item[training_set[i,2]]^2 + bias_user[training_set[i,1]]^2 + norm(q[training_set[i,2],:])^2 + 
                          norm(p[training_set[i,1],:])^2) 
        end
    end

    #Previsão do conjunto de teste
    RSME = root_squared_mean_error_IRSVD(test_set,bias_item,bias_user,mean_global_train,q,p)

    return RSME
end

####################################################################################################################################

#Realização dos testes (inclusos no relatório)

#Número de "variáveis latentes"(dimensão ou fatores) que serão usados na computação da previsão.
num_of_factors = 100 

#Inicializando 'q'
q = rand(qtd_items,num_of_factors) #Item

#Inicializando 'p'
p = rand(qtd_users,num_of_factors) #Usuários

#Cópia de "p" e "q"
q_copy = copy(q)
p_copy = copy(p)

qtd_factors = 10
Resultados1 = zeros(10,2)
Resultados2 = zeros(10,2)

#Critério de parada utilizado: 0.1
for i=1:10
    @time Resultados1[i,1] = Regulared_SVD(train1,test1,0.02,0.001,0.1,q[:,1:qtd_factors],p[:,1:qtd_factors])
    p = copy(p_copy)
    q = copy(q_copy)
    @time Resultados2[i,1] = Regulared_SVD(train2,test2,0.02,0.001,0.1,q[:,1:qtd_factors],p[:,1:qtd_factors])
    p = copy(p_copy)
    q = copy(q_copy)
    @time Resultados1[i,2] = Improved_Regulared_SVD(train1,test1,0.02,0.005,0.1,q[:,1:qtd_factors],p[:,1:qtd_factors])
    p = copy(p_copy)
    q = copy(q_copy)
    @time Resultados2[i,2] = Improved_Regulared_SVD(train2,test2,0.02,0.005,0.1,q[:,1:qtd_factors],p[:,1:qtd_factors])
    p = copy(p_copy)
    q = copy(q_copy)
    qtd_factors += 10
end

#println(Resultados1)
#println(Resultados2)
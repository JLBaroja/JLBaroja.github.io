# Inferencia con JAGS

Como muestra el ejemplo de inferencia en el cuadernillo anterior (_04_Distribuciones_Continuas_), al trabajar con variables no-observables continuas es necesario integrar ciertas funciones para calcular la incertidumbre posterior sobre las mismas. En el ejercicio precedente esa integración es "amigable", o al menos "tratable", lo cual quiere decir que es posible calcular las integrales analíticamente. Sin embargo en muchos escenarios aplicados resolver esas integrales resulta complicado y en algunos casos imposible, o al menos imposible respecto de las técnicas de integración analítica actuales.

Por fortuna, desde el nacimiento del cálculo infinitesimal se han desarrollado varios algoritmos para _aproximar_ la solución a integrales intratables, y los avances en computación de las últimas décadas han hecho posible resolver integrales complejas en poco tiempo y con alta precisión. 

Además, muchos de esos programas y algoritmos han sido específicamente diseñados para implementar inferencia Bayesiana. Existen muchos en diferentes lenguajes (e.g., `Stan`, `PyMC`, `OpenBUGS`, `Mamba`) aparte de `JAGS` (_Just Another Gibbs Sampler_; Plummer, 2003), que es el que usaremos en este curso.

### Solución por Métodos Numéricos (inferencia con JAGS)

A continuación mostraremos cómo resolver el ejercicio del apunte anterior en `JAGS`. Recapitulando, el modelo con el que trabajaremos es:

$$\begin{align}
    \theta&\sim Beta(1,2)\\    
    r&\leftarrow 5\\
    x&\leftarrow 2\\
    x|\theta,r&\sim NegBinomial(\theta,r),\\
    \end{align}$$
    
<center><img src='neg_bin_model_tex_2.png' width=200 height=200 /></center>

con solución analítica:

$$\begin{align}
p(\theta|x=2)&=\frac{p(x=2|\theta)p(\theta)}{\int_\theta p(x=2|\theta)p(\theta)}\\\\
             &=\frac{(15\theta^5-30\theta^6+15\theta^7)(2-2\theta)}{\int_{0}^{1}(15\theta^5-30\theta^6+15\theta^7)(2-2\theta)d\theta}\\\\
             &=\frac{-30\theta^8+90\theta^7-90\theta^6+30\theta^5}{\frac{10}{168}}\\\\
             &=-504\theta^8+1512\theta^7-1512\theta^6+504\theta^5
\end{align}$$

El script siguiente implementa este modelo en `JAGS`:


```R
### Calculando POSTERIORES con JAGS

## Variables Observadas
# En esta sección tenemos objetos de R que contienen las observaciones/datos disponibles:
x <- 2
r <- 5
observed <- list('x','r') # Es necesario incluir los *nombres* de esos objetos en esta lista

## Variables No-observables
# Sección para especificar *cómo se llaman* los no observables en el modelo
unobserved <- c('theta') # Noten que esta no es una lista, sino una arreglo (o vector)

## Modelo
# Especificación del *modelo gráfico* en lenguaje BUGS
# ("each arrow in the graphical model is a line of code" -J. Kruschke)
write('

model{
    # Incertidumbre inicial sobre no-observables
    theta~dbeta(1,2)
    # Verosimilitud
    x~dnegbin(theta,r) # JAGS/BUGS incluye muchas distribuciones conocidas (ver manual)
}

','jags_example.bug')

## Algoritmo
# "Magia" de JAGS: con la información de las observadas, la incertidumbre
# inicial sobre las no-observadas, y el modelo que las relaciona,
# JAGS genera "muestras" de la distribución posterior correspondiente
# ¡y nos ahorra las integrales!
library('R2jags')
bayes <- jags(data=observed, # LISTA definida en línea 7
             parameters.to.save=unobserved, # ARREGLO definido en línea 11
             model.file='jags_example.bug') # MODELO definido en líneas 16-25

## Distribuciones
# Aparte de guardar las muestras de la posterior, el objeto 'bayes' contiene
# mucha información sobre el resultado del algoritmo. Si bien dicha información
# es muy importante para evaluar la calidad del resultado, por el momento saltaremos
# a examinarlo visualmente. Para ello es conveniente guardar la sección de 'bayes'
# que tiene las posteriores en un objeto nuevo:
nodos <- bayes$BUGSoutput$sims.list

## Gráficas
# Para examinar la distribución posterior generada por JAGS
# basta con hacer un histograma de la sección correspondiente 
# en 'nodos'. 
# Aparte, en este ejemplo incluimos la solción analítica aprovechando
# que la tenemos disponible, para compararla contra la solución de
# JAGS.
col_post <- '#ec102f'
options(repr.plot.width = 9, repr.plot.height = 7)
par(mar=c(4.5,4,3,1))
hist(nodos$theta,
     freq = F,breaks=100,
     axes=F,ann=F,
     xlim=c(0,1),ylim=c(0,3),
     col=paste(col_post,'88',sep=''),border=F)
theta_analitic <- seq(0,1,length.out = 100)
post_analitic <- -504*theta_analitic^8+1512*theta_analitic^7-1512*theta_analitic^6+504*theta_analitic^5
lines(theta_analitic,post_analitic,lwd=5,col=col_post)
legend(0.05,3,pch=c(15,NA),lwd=c(0,5),legend=c('solución numérica (JAGS)','solución analítica'),
      pt.cex=2,col=c(paste(col_post,'88',sep=''),col_post),cex=1.3,seg.len=1,bg='#00000000',
      x.intersp=.5,box.lty='blank')
axis(1,cex.axis=1.75,lwd=2,padj=.25)
axis(2,las=1,hadj=.75)
mtext('densidad',2,line=2.5,cex=1.5)
mtext('\u03b8',1,line=3.5,cex=2.5)
```

    Loading required package: rjags
    
    Loading required package: coda
    
    Linked to JAGS 4.3.0
    
    Loaded modules: basemod,bugs
    
    
    Attaching package: ‘R2jags’
    
    
    The following object is masked from ‘package:coda’:
    
        traceplot
    
    
    module glm loaded
    

    Compiling model graph
       Resolving undeclared variables
       Allocating nodes
    Graph information:
       Observed stochastic nodes: 1
       Unobserved stochastic nodes: 1
       Total graph size: 5
    
    Initializing model
    



    
![png](output_1_2.png)
    



```R
head(nodos$theta)
```


<table>
<caption>A matrix: 6 × 1 of type dbl</caption>
<tbody>
	<tr><td>0.3637828</td></tr>
	<tr><td>0.6911957</td></tr>
	<tr><td>0.7843047</td></tr>
	<tr><td>0.7429606</td></tr>
	<tr><td>0.7067321</td></tr>
	<tr><td>0.7089708</td></tr>
</tbody>
</table>



Resumiendo, la gran ventaja de `JAGS` es que una vez que tenemos el modelo expresado en notación gráfica es fácil traducirlo a lenguaje `BUGS` y dejar que `JAGS` calcule la posterior de interés.

Sin embargo hay algunas desventajas de utilizar `JAGS`. Quizá la más importante es que, por default, **no** devuelve muestras de la distribución a priori, que es una distribución crucial para interpretar la posterior correspondiente en tanto que ambas distribuciones representan estados de conocimiento igualmente importantes sobre los no-observables en un modelo. Afortunadamente es posible hacer que el programa también devuelva la distribución a priori sobre la(s) variable(s) de interés: para ello es necesario "sacarle una copia" al modelo en `BUGS`, pero omitiendo las conexiones entre las no-observadas en esa copia con las variables observadas. 

Esta técnica se presenta en el script siguiente, sobre el modelo/ejemplo anterior:


```R
### Calculando PRIORS y POSTERIORES con JAGS

## Variables Observadas
# En esta sección tenemos objetos de R que contienen las observaciones/datos disponibles:
x <- 2
r <- 5
observed <- list('x','r') # Es necesario incluir los *nombres* de esos objetos en esta lista

## Variables No-observables
# Sección para especificar *cómo se llaman* los no observables en el modelo
unobserved <- c('theta_prior','theta_post') # En general este arreglo contiene los nombres
# de todo lo que queremos que JAGS rastree (priors y posteriores)

## Modelo
# Especificación del *modelo gráfico* en lenguaje BUGS
# ("each arrow in the graphical model is a line of code" -J. Kruschke)
write('

model{
    ## Copia que calcula POSTERIORES
    # Incertidumbre inicial sobre no-observables
    theta_post~dbeta(1,2)
    # Verosimilitud
    x~dnegbin(theta_post,r) # Noten el cambio de nombre *theta_post*

    ## Copia que calcula PRIORS
    # Incertidumbre inicial sobre no-observables
    theta_prior~dbeta(1,2) # Para calcular el prior sobre theta
                           # es necesario que *theta_prior* tengab
                           # *exactamente* la misma definición
                           # que *theta_post*.
}

','jags_example_2.bug') # Noten el cambio de nombre del modelo

## Algoritmo
# "Magia" de JAGS: con la información de las observadas, la incertidumbre
# inicial sobre las no-observadas, y el modelo que las relaciona,
# JAGS genera "muestras" de la distribución posterior correspondiente
# ¡y nos ahorra las integrales!
library('R2jags')
bayes2 <- jags(data=observed, # LISTA definida en línea 7
             parameters.to.save=unobserved, # ARREGLO definido en línea 11
             model.file='jags_example_2.bug') # MODELO definido en líneas 17-34

## Distribuciones
nodos2 <- bayes2$BUGSoutput$sims.list # Noten el cambio de nombre 'bayes2' y 'nodos2':
                                     # es conveniente guardar los resultados
                                     # de distintos modelos en diferentes objetos en R

## Gráficas
col_prior <- '#f06614'
col_post <- '#ec102f'
options(repr.plot.width = 9, repr.plot.height = 7)
par(mar=c(4.5,4,3,1))
hist(nodos2$theta_prior,
     freq = F,breaks=100,
     axes=F,ann=F,
     xlim=c(0,1),ylim=c(0,3),
     col=paste(col_prior,'77',sep=''),border=F)
hist(nodos2$theta_post,
     freq = F,breaks=100,
     axes=F,ann=F,add=T,
     xlim=c(0,1),ylim=c(0,3),
     col=paste(col_post,'55',sep=''),border=F)
theta_analitic <- seq(0,1,length.out = 100)
prior_analitic <- 2-2*theta_analitic
post_analitic <- -504*theta_analitic^8+1512*theta_analitic^7-1512*theta_analitic^6+504*theta_analitic^5
lines(theta_analitic,prior_analitic,lwd=6,col=col_prior)
lines(theta_analitic,post_analitic,lwd=6,col=col_post)
legend(0.05,3,pch=22,pt.lwd=3,lwd=0,legend=c('prior','posterior'),
      pt.cex=2.5,col=c(col_prior,col_post),pt.bg=paste(c(col_prior,col_post),'44',sep=''),
      cex=1.3,bg='#00000000',
      x.intersp=0,box.lty='blank')
axis(1,cex.axis=1.75,lwd=2,padj=.25)
axis(2,las=1,hadj=.75)
mtext('densidad',2,line=2.5,cex=1.5)
mtext('\u03b8',1,line=3.5,cex=2.5)
```

    Compiling model graph
       Resolving undeclared variables
       Allocating nodes
    Graph information:
       Observed stochastic nodes: 1
       Unobserved stochastic nodes: 2
       Total graph size: 6
    
    Initializing model
    



    
![png](output_4_1.png)
    


El resultado de `JAGS` es "imperfecto" por tratarse de una aproximación numérica: siempre habrá cierta discordancia entre el resultado analítico y el que encuentra el algoritmo, aunque ambas soluciones generalmente coinciden en las conclusiones generales.

Existen variables y argumentos en el algoritmo para mejorar la precisión de las distribuciones que devuelve. Los más importantes se incluyen en el script siguiente, en las líneas `45-48`. El **número de cadenas** `n.chains` es el número de veces que `JAGS` intentará recuperar las distribuciones de interés: la idea es comenzar cada cadena en diferentes valores de $\theta$ y examinar el grado de acuerdo entre cadenas: si todas las cadenas llegan a la misma conclusión sin importar dónde empieza cada una, el resultado es confiable. El **número de iteraciones** `n.iter` totales indica cuántas muestras de las distribuciones compondrán cada cadena: los histogramas que hemos generado están compuestos por largas cadenas de valores posibles de $\theta$ que al graficarse en ese formato forman las curvas que buscamos: la longitud de esas cadenas depende de este argumento, junto con otros dos: el **número de burn-in** `n.burnin` es la cantidad de valores descartados al inicio de cada cadena: la idea es eliminar las "imperfecciones" del inicio del algoritmo y sólo conservar los valores que "ya llegaron" a la distribución de interés; mientras que la **fineza** `n.thin` indica cada cuántos valores generados por el algoritmo guardamos uno en cada cadena.


```R
### Calculando PRIORS y POSTERIORES con JAGS

## Variables Observadas
# En esta sección tenemos objetos de R que contienen las observaciones/datos disponibles:
x <- 2
r <- 5
observed <- list('x','r') # Es necesario incluir los *nombres* de esos objetos en esta lista

## Variables No-observables
# Sección para especificar *cómo se llaman* los no observables en el modelo
unobserved <- c('theta_prior','theta_post') # En general este arreglo contiene los nombres
# de todo lo que queremos que JAGS rastree (priors y posteriores)

## Modelo
# Especificación del *modelo gráfico* en lenguaje BUGS
# ("each arrow in the graphical model is a line of code" -J. Kruschke)
write('

model{
    ## Copia que calcula POSTERIORES
    # Incertidumbre inicial sobre no-observables
    theta_post~dbeta(1,2)
    # Verosimilitud
    x~dnegbin(theta_post,r) # Noten el cambio de nombre *theta_post*

    ## Copia que calcula PRIORS
    # Incertidumbre inicial sobre no-observables
    theta_prior~dbeta(1,2) # Para calcular el prior sobre theta
                           # es necesario que *theta_prior* tenga
                           # *exactamente* la misma definición
                           # que *theta_post*.
}

','jags_example_3.bug') # Noten el cambio de nombre del modelo

## Algoritmo
# "Magia" de JAGS: con la información de las observadas, la incertidumbre
# inicial sobre las no-observadas, y el modelo que las relaciona,
# JAGS genera "muestras" de la distribución posterior correspondiente
# ¡y nos ahorra las integrales!
library('R2jags')
bayes3 <- jags(data=observed, # LISTA definida en línea 7
             parameters.to.save=unobserved, # ARREGLO definido en línea 11
             model.file='jags_example_3.bug', # MODELO definido en líneas 17-34
             n.chains=3, # Número de cadenas (¿cuántas veces quieres aproximar la misma posterior?)
             n.burnin=2000, # Número de muestras descartadas contando desde el inicio del proceso
             n.thin=1, # Cada cuántas interaciones guardamos una muestra en el resultado final
             n.iter=10000) # Cuántas iteraciones en total pedimos por cadena

## Distribuciones
nodos3 <- bayes3$BUGSoutput$sims.list # Noten el cambio de nombre 'bayes2' y 'nodos2':
                                     # es conveniente guardar los resultados
                                     # de distintos modelos en diferentes objetos en R

## Gráficas
col_prior <- '#f06614'
col_post <- '#ec102f'
options(repr.plot.width = 9, repr.plot.height = 7)
par(mar=c(4.5,4,3,1))
hist(nodos3$theta_prior,
     freq = F,breaks=100,
     axes=F,ann=F,
     xlim=c(0,1),ylim=c(0,3),
     col=paste(col_prior,'77',sep=''),border=F)
hist(nodos3$theta_post,
     freq = F,breaks=100,
     axes=F,ann=F,add=T,
     xlim=c(0,1),ylim=c(0,3),
     col=paste(col_post,'55',sep=''),border=F)
theta_analitic <- seq(0,1,length.out = 100)
prior_analitic <- 2-2*theta_analitic
post_analitic <- -504*theta_analitic^8+1512*theta_analitic^7-1512*theta_analitic^6+504*theta_analitic^5
lines(theta_analitic,prior_analitic,lwd=6,col=col_prior)
lines(theta_analitic,post_analitic,lwd=6,col=col_post)
legend(0.05,3,pch=22,pt.lwd=3,lwd=0,legend=c('prior','posterior'),
      pt.cex=2.5,col=c(col_prior,col_post),pt.bg=paste(c(col_prior,col_post),'44',sep=''),
      cex=1.3,bg='#00000000',
      x.intersp=0,box.lty='blank')
axis(1,cex.axis=1.75,lwd=2,padj=.25)
axis(2,las=1,hadj=.75)
mtext('densidad',2,line=2.5,cex=1.5)
mtext('\u03b8',1,line=3.5,cex=2.5)
```

    Compiling model graph
       Resolving undeclared variables
       Allocating nodes
    Graph information:
       Observed stochastic nodes: 1
       Unobserved stochastic nodes: 2
       Total graph size: 6
    
    Initializing model
    



    
![png](output_6_1.png)
    


En este problema de ejemplo, que es relativamente sencillo, el algoritmo tiene pocos problemas y puede rastrear la solución casi de inmediato. Sin embargo, en problemas más complejos que involucran muchas variables observadas y no-observadas, así como relaciones más intrincadas entre ellas, es común que `JAGS` se "atore" buscando la solución. En estos casos podemos ayudarle al algormito a encontrar la solución manipulando los argumentos recién explicados.

Y, aparte, siempre podemos (y debemos) evaluar la calidad de la solución de `JAGS`, en tanto que ésta puede estar poco o muy lejos de la solución analítica objetivo. Para hacerlo existen dos métodos principales, el primero es la **inspección visual de las cadenas**, y el segundo es el **valor de ciertos estadísticos** que resumen el comportamiento de las cadenas.

Para examinar las cadenas visualmente:


```R
options(repr.plot.width = 15, repr.plot.height = 6)
traceplot(bayes3) # Cambiar a 'bayes2' para comparar las diferencias
```


    
![png](output_8_0.png)
    



    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    


Gráficamente, las cadenas que se ven como una **oruga peluda** (Lee & Wagenmakers, 2013) reflejan un resultado confiable, porque esa apariencia indica que cada cadena alcanzó la misma solución de manera independiente. En cambio, cuando es posible distinguir e identificar a cada cadena por separado tenemos razones para cuestionar el resultado del algoritmo, porque en ese escenario cada cadena apoya una solución diferente. Cuando eso ocurre es _necesario_ repetir el análisis cambiando los valores de los argumentos del algoritmo hasta conseguir la oruga peluda que indique buena **convergencia**.

No siempre es posible o práctico examinar visualmente las cadenas de _todas_ las variables de un modelo (generalmente trabajaremos con modelos de varios cientos o miles de variables). En estos casos conviene revisar un par de estadísticos que resumen el comportamiento de las cadenas en cada variable. Dichos estadísticos se conocen como $\hat{R}$ y **n.eff**, y pueden consultarse en el resultado que el algoritmo devuelve utilizando la instrucción:


```R
bayes3$BUGSoutput$summary
```


<table>
<caption>A matrix: 3 × 9 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>2.5%</th><th scope=col>25%</th><th scope=col>50%</th><th scope=col>75%</th><th scope=col>97.5%</th><th scope=col>Rhat</th><th scope=col>n.eff</th></tr>
</thead>
<tbody>
	<tr><th scope=row>deviance</th><td>4.0377663</td><td>1.4713196</td><td>2.96076128</td><td>3.0742243</td><td>3.4736965</td><td>4.4174774</td><td>8.1678580</td><td>1.001005</td><td>24000</td></tr>
	<tr><th scope=row>theta_post</th><td>0.5984345</td><td>0.1482182</td><td>0.29614079</td><td>0.4967386</td><td>0.6044767</td><td>0.7088702</td><td>0.8621215</td><td>1.001049</td><td>18000</td></tr>
	<tr><th scope=row>theta_prior</th><td>0.3341153</td><td>0.2357721</td><td>0.01278777</td><td>0.1349683</td><td>0.2937548</td><td>0.4996404</td><td>0.8436378</td><td>1.001035</td><td>20000</td></tr>
</tbody>
</table>



Aparte de regresar los dos estadísticos de interés, en las dos columnas finales, dicha instrucción devuelve algunos descriptivos de la distribución correspondiente, como la ubicación central y medidas intercuartilares. Respecto de los estadísticos de calidad, si las cadenas **convergen** el estadístico debe ser $\hat{R}\leq 1.05$, y `n.eff` debe ser lo más cercano posible a `(n.iter-n.burnin)*n.chains/n.thin`.

### Ejercicio

1. Instalar `JAGS` (http://mcmc-jags.sourceforge.net/) y vincularlo con `R`: después de bajar los ejecutables de `JAGS` e instalarlos en el equipo **es necesario** instalar la librería de `R` que vincula ambos programas. Para hacerlo pueden ejecutar la instrucción `install.packages('R2jags')` directamente en la consola de `R` o de `RStudio`.

2. Correr cualquiera de los 3 ejemplos anteriores y **asegurarse** de reproducir los resultados de este notebook: si los histogramas aparecen como se han presentado en este apunte pueden estar seguras de que `JAGS` ha sido instalado correctamente y que está vinculado con `R`. 

3. Mandar una captura de pantalla que compruebe la instalación exitosa.

### Referencias

* Plummer, M. (2003). JAGS: A program for analysis of Bayesian graphical models using Gibbs sampling. In K. Hornik, F. Leisch, & A. Zeileis (Eds.), _Proceedings of the 3rd International Workshop on Distributed Statistical Computing._ Vienna, Austria. 
* Plummer, M. (2017). _JAGS Version 4.3.0 User Manual_: https://people.stat.sc.edu/hansont/stat740/jags_user_manual.pdf 

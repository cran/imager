% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{correlate}
\alias{correlate}
\alias{convolve}
\title{Correlation/convolution of image by filter}
\usage{
correlate(im, filter, dirichlet = TRUE, normalise = FALSE)

convolve(im, filter, dirichlet = TRUE, normalise = FALSE)
}
\arguments{
\item{im}{an image}

\item{filter}{the correlation kernel.}

\item{dirichlet}{boundary condition. Dirichlet if true, Neumann if false (default TRUE, Dirichlet)}

\item{normalise}{compute a normalised correlation (ie. local cosine similarity)}
}
\description{
The correlation of image im by filter flt is defined as:
 \eqn{res(x,y,z) = sum_{i,j,k} im(x + i,y + j,z + k)*flt(i,j,k).}
 The convolution of an image img by filter flt is defined to be:
      \eqn{res(x,y,z) = sum_{i,j,k} img(x-i,y-j,z-k)*flt(i,j,k)}
}
\section{Functions}{
\itemize{
\item \code{convolve()}: convolve image with filter

}}
\examples{
\dontshow{cimg.limit.openmp()}
#Edge filter
filter <- as.cimg(function(x,y) sign(x-5),10,10) 
layout(t(1:2))
#Convolution vs. correlation 
correlate(boats,filter) \%>\% plot(main="Correlation")
convolve(boats,filter) \%>\% plot(main="Convolution")
}

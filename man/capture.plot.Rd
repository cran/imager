% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/cimg_class.R
\name{capture.plot}
\alias{capture.plot}
\title{Capture the current R plot device as a cimg image}
\usage{
capture.plot()
}
\value{
a cimg image corresponding to the contents of the current plotting window
}
\description{
Capture the current R plot device as a cimg image
}
\examples{
\dontshow{cimg.limit.openmp()}
##interactive only:
##plot(1:10)
###Make a plot of the plot
##capture.plot() \%>\% plot 
}
\author{
Simon Barthelme
}

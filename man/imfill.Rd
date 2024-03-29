% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/imgen.R
\name{imfill}
\alias{imfill}
\title{Create an image of custom size by filling in repeated values}
\usage{
imfill(x = 1, y = 1, z = 1, val = 0, dim = NULL)
}
\arguments{
\item{x}{width (default 1)}

\item{y}{height (default 1)}

\item{z}{depth (default 1)}

\item{val}{fill-in values. Either a single value (for grayscale), or RGB values for colour, or a character string for a colour (e.g. "blue")}

\item{dim}{dimension vector (optional, alternative to specifying x,y,z)}
}
\value{
an image object (class cimg)
}
\description{
This is a convenience function for quickly creating blank images, or images filled with a specific colour. See examples.
If val is a logical value, creates a pixset instead.
}
\examples{
\dontshow{cimg.limit.openmp()}

imfill(20,20) \%>\% plot #Blank image of size 20x20
imfill(20,20,val=c(1,0,0)) \%>\% plot #All red image
imfill(20,20,val="red") \%>\% plot #Same, using R colour name
imfill(3,3,val=FALSE) #Pixset
imfill(dim=dim(boats)) #Blank image of the same size as the boats image
}
\author{
Simon Barthelme
}

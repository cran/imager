% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/loading.R
\name{load.video}
\alias{load.video}
\title{Load a video using ffmpeg}
\usage{
load.video(
  fname,
  maxSize = 1,
  skip.to = 0,
  frames = NULL,
  fps = NULL,
  extra.args = "",
  verbose = FALSE
)
}
\arguments{
\item{fname}{file to load}

\item{maxSize}{max. allowed size in memory, in GB (default max 1GB).}

\item{skip.to}{skip to a certain point in time (in sec., or "hh:mm::ss" format)}

\item{frames}{number of frames to load (default NULL, all)}

\item{fps}{frames per second (default NULL, determined automatically)}

\item{extra.args}{extra arguments to be passed to ffmpeg (default "", none)}

\item{verbose}{if TRUE, show ffmpeg output (default FALSE)}
}
\value{
an image with the extracted frames along the "z" coordinates
}
\description{
You need to have ffmpeg on your path for this to work. This function uses ffmpeg to split the video into individual frames, which are then loaded as images and recombined.
Videos are memory-intensive, and load.video performs a safety check before loading a video that would be larger than maxSize in memory (default 1GB)
}
\examples{
\dontshow{cimg.limit.openmp()}

fname <- system.file('extdata/tennis_sif.mpeg',package='imager')
##Not run
## load.video(fname) \%>\% play
## load.video(fname,fps=10) \%>\% play
## load.video(fname,skip=2) \%>\% play
}
\seealso{
save.video, make.video
}
\author{
Simon Barthelme
}

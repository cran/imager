% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/loading.R
\name{make.video}
\alias{make.video}
\alias{save.video}
\title{Make/save a video using ffmpeg}
\usage{
make.video(
  dname,
  fname,
  pattern = "image-\%d.png",
  fps = 25,
  extra.args = "",
  verbose = FALSE
)

save.video(im, fname, ...)
}
\arguments{
\item{dname}{name of a directory containing individual files}

\item{fname}{name of the output file. The format is determined automatically from the name (example "a.mpeg" will have MPEG format)}

\item{pattern}{pattern of filename for frames (the default matches "image-1.png", "image-2.png", etc.. See ffmpeg documentation for more).}

\item{fps}{frames per second (default 25)}

\item{extra.args}{extra arguments to be passed to ffmpeg (default "", none)}

\item{verbose}{if TRUE, show ffmpeg output (default FALSE)}

\item{im}{an image or image list}

\item{...}{extra arguments to save.video, passed on to make.video}
}
\description{
You need to have ffmpeg on your path for this to work. This function uses ffmpeg to combine individual frames into a video.
save.video can be called directly with an image or image list as input.
make.video takes as argument a directory that contains a sequence of images representing individual frames to be combined into a video.
}
\section{Functions}{
\itemize{
\item \code{save.video()}: Save a video using ffmpeg

}}
\examples{
\dontshow{cimg.limit.openmp()}
## Not run
## iml <- map_il(seq(0,20,l=60),~ isoblur(boats,.))
## f <- tempfile(fileext=".avi")
## save.video(iml,f)
## load.video(f) \%>\% play
## #Making a video from a directory
## dd <- tempdir()
## for (i in 1:length(iml)) {
## png(sprintf("\%s/image-\%i.png",dd,i));
## plot(iml[[i]]); dev.off() }
## make.video(dd,f)
## load.video(f) \%>\% play
}
\seealso{
load.video
}
\author{
Simon Barthelme
}

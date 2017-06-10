#install.packages("ggplot2")
#install.packages("dplyr")
#setwd("~/phonemes/src")
library(ggplot2)
library(dplyr)
library(reshape2)
repr.names <- c("MFCC", "Conv", "Rec1", "Rec2", "Rec3", "Rec4", "Rec5")
data <- read.table("err_boot.csv", 
                  col.names = repr.names)

ggplot(melt(data), aes(x=variable, y=value)) + geom_boxplot(notch = TRUE) + 
  xlab("Representation") +
  ylab("Error rate") +
  theme(aspect.ratio=3/4, text=element_text(size=22))

ggsave("../figures/decode.pdf", width=8, height=10)

abx <- read.table("abx_cv_scores.txt", header=TRUE) %>% 
  mutate(correct = score >= 0,
         repr=factor(repr, levels=c("mfcc","conv","rec1","rec2","rec3","rec4","rec5"))) %>%
  rename(Representation=repr)



abx_same <- abx %>% dplyr::filter(target_c == distractor_c) %>% rename(Class=target_c)
se <- function(x) sd(x)/sqrt(length(x))
ci <- function(x) se(x) * qt(0.95/2 + 0.5, length(x)-1)

ggplot(abx_same %>% group_by(Representation, Class) %>% summarize(Accuracy=mean(correct), SE=se(correct), CI=ci(correct)),
       aes(x=Representation, y=Accuracy, color=Class, group=Class)) + 
  geom_ribbon(aes(ymin=Accuracy-SE, ymax=Accuracy+SE, fill=Class, color=NULL), alpha=0.1) +
  geom_point(size=3) + 
  geom_line(alpha=0.5, size=2) +
  theme(aspect.ratio=1, text=element_text(size=22), legend.position = "bottom")
ggsave("../figures/abx_cv_same.pdf", width=8, height=10)


syn.in <- read.csv("synonyms_errorrates.csv")
syn <- melt(syn.in) %>% rename(Pair=variable, Error=value, Representation=repr) %>%
  mutate(Representation=factor(Representation, levels=c("mfcc","conv","rec1","rec2", "rec3", "rec4","rec5","emb")))
head(syn)

ggplot(syn, aes(x=Representation, y=Error, color=Pair, group=Pair)) + geom_point(size=3) + geom_line(alpha=0.5, size=2) +
  theme(text=element_text(size=22), aspect.ratio=1, legend.position="bottom", legend.direction = "vertical")
ggsave(filename = "../figures/synonym.pdf", width=10, height=12)

data_cor = read.table(header=TRUE, text="
Representation r
conv 0.95 
rec1 0.87
rec2 0.81
rec3 0.76
rec4 0.71
rec5 0.68
")
ggplot(data_cor, aes(x=Representation, y=r, group=1)) + geom_point(size=3) + geom_line(alpha=0.5, size=2) +
  theme(text=element_text(size=22), aspect.ratio=1/3) + ylab("Pearson's r")
ggsave(filename = "../figures/correlation_mfcc.pdf", width=10, height=4)

data_hier <- read.table(header=TRUE, text="
Representation ARI
mfcc 0.27 
conv 0.12
rec1 0.23
rec2 0.16
rec3 0.16
rec4 0.15
rec5 0.15
    ")
ggplot(data_hier %>% mutate(Representation=factor(Representation, levels=c("mfcc","conv","rec1","rec2", "rec3", "rec4","rec5"))), aes(x=Representation, y=ARI, group=1)) + geom_point(size=3) + geom_line(alpha=0.5, size=2) +
  theme(text=element_text(size=22), aspect.ratio=1/3) + ylab("Adjusted Rand Index")
ggsave(filename = "../figures/hier_ari.pdf", width=10, height=4)

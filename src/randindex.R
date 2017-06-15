library(dplyr)
library(ggplot2)
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

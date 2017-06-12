library(ggplot2)
library(dplyr)
library(reshape2)

syn <- read.table("synonym_scores.txt", header=TRUE) %>%
  mutate(Representation=factor(Representation, levels=c("mfcc","conv","rec1","rec2", "rec3", "rec4","rec5","emb")))

lev <- (syn %>% filter(Representation=="emb") %>% group_by(Pair) %>% summarize(Error=mean(Error)) %>% arrange(Error))$Pair
syn <- syn %>% mutate(Pair=factor(Pair, levels=lev))


head(syn)

ggplot(syn, aes(x=Representation, y=Error, color=Pair, group=Pair)) + geom_point(size=3) + geom_line(alpha=0.5, size=2) +
  theme(text=element_text(size=22), aspect.ratio=1, legend.position="bottom", legend.direction = "vertical")
ggsave(filename = "../figures/synonym.pdf", width=10, height=12)

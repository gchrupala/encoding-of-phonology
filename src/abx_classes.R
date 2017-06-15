library(dplyr)
library(ggplot2)

abx <- read.table("abx_cv_scores.txt", header=TRUE) %>% 
  mutate(correct = score >= 0,
         repr=factor(repr, levels=c("mfcc","conv","rec1","rec2","rec3","rec4","rec5","emb"))) %>%
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



# Defining labor markets using Natural Language Processing (NLP) and task descriptions

## About this repository

The purpose of this project is to improve upon existing definitions of labor markets by using methods from Natural Language Processing (NLP). I define labor markets in a novel way by clustering occupations based on their task descriptions.

**Contents**

## Background information

At the core of many papers in labour economics is the definition of the labour market. For example,
in a seminal work on the effects of immigration, Borjas (2003) argued that labour markets are
appropriately defined as groups of workers with similar education and experience. Borjas’
conclusions about the effects of immigration in that paper depend heavily on the labour market
definition. Other studies in the immigration literature (e.g., Ottaviano et al., 2016) implicitly define
labour markets at the industry-year level. However, it is possible that true labour market
boundaries do not perfectly overlap with standard industry classification. It is possible that
researchers arrive at different conclusions in their studies if they adopt different definitions of the
labour market in their analyses.

Researchers that study imperfect competition in labour markets must also choose a definition of
the labour market in their studies. For example, several authors studying labour market
concentration have chosen to define labour markets as a combination of commuting zone and
occupation/industry (Azar et al., 2019; Azar, Marinescu, & Steinbaum, 2020; Azar, Marinescu,
Steinbaum, et al., 2020; Qiu & Sojourner, 2019; Rinz, 2018). However, these labour market
definitions are limited because workers often switch occupations and/or industries through-out
their career. Narrow definitions of the labour market will tend to overstate the degree of employer
concentration in labour markets. Recognizing this, some recent studies on labour market
concentration incorporate worker flows across occupations and/or industries in their definition of
the labour market (Arnold, 2019; Schubert et al., 2020).

There are many other examples of studies in the labour economics literature that depend crucially
on the definition of the labour market. For example, Autor et al. (2013) use variation between local
labor markets to study the effects of global trade shocks. Labour market definitions are also
relevant for the study of a variety of policies, such as the effect of unemployment insurance on the
non-eligible (Lalive et al., 2015).

In this paper, I use machine learning methods to define labour markets. Specifically, I use stateof-
the-art methods from Natural Language Processing (NLP) to cluster occupations based on their
task descriptions. I plan to refine these results by incorporating education and skill requirements.
Eventually, the results will be used to study labour market concentration, imperfect competition in
labour markets, and immigration.

My ultimate goal is to define a ``labor market”. Specifically in my context, a ``labor market” is a
cluster of occupations for which (a) the tasks of all of the occupations are similar and (b) the
workers of these occupations have similar abilities, education, and experience. This is useful for
researchers with datasets that include occupation codes for workers. Two workers in the same ``labor market” should be very substitutable, even if they have different occupation codes in the
data.


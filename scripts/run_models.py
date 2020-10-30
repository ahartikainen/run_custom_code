import stan

schools_code = """
data {
  int<lower=0> J;         // number of schools
  real y[J];              // estimated treatment effects
  real<lower=0> sigma[J]; // standard error of effect estimates
}
parameters {
  real mu;                // population treatment effect
  real<lower=0> tau;      // standard deviation in treatment effects
  vector[J] eta;          // unscaled deviation from mu by school
}
transformed parameters {
  vector[J] theta = mu + tau * eta;        // school treatment effects
}
model {
  target += normal_lpdf(eta | 0, 1);       // prior log-density
  target += normal_lpdf(y | theta, sigma); // log-likelihood
}
"""


def model_1():
    print("RUNNING MODEL 1")
    schools_data = {
        "J": 8,
        "y": [28, 8, -3, 7, -1, 1, 18, 12],
        "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
    }

    posterior = stan.build(schools_code, data=schools_data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    print(fit)


def model_2():
    print("RUNNING MODEL 2")
    schools_data = {
        "J": 8 * 10,
        "y": [28, 8, -3, 7, -1, 1, 18, 12] * 10,
        "sigma": [15, 10, 16, 11, 9, 11, 10, 18] * 10,
    }

    posterior = stan.build(schools_code, data=schools_data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    print(fit)


def model_3():
    print("RUNNING MODEL 3")
    schools_data = {
        "J": 8 * 11,
        "y": [28, 8, -3, 7, -1, 1, 18, 12] * 11,
        "sigma": [15, 10, 16, 11, 9, 11, 10, 18] * 11,
    }

    posterior = stan.build(schools_code, data=schools_data)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    print(fit)


if __name__ == "__main__":
    try:
        model_1()
    except:
        print("FAILURE 1")
    try:
        model_2()
    except:
        print("FAILURE 2")
    try:
        model_3()
    except:
        print("FAILURE 3")

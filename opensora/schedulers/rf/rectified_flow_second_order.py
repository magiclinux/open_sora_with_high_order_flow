import torch
from torch.distributions import LogisticNormal

from ..iddpm.gaussian_diffusion import _extract_into_tensor, mean_flat

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if model_kwargs["num_frames"][0] == 1:
        num_frames = torch.ones_like(model_kwargs["num_frames"])
    else:
        num_frames = model_kwargs["num_frames"] // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler_Second_Order:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale
        
    def training_losses_second_order(self, model, second_order_model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        x_t, first_order_gt, second_order_gt = self.add_noise_first_and_second_order(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0, x_t0_first_order_gt, x_t0_second_order_gt = self.add_noise_first_and_second_order(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)
            first_order_gt = torch.where(mask[:, None, :, None, None], first_order_gt, x_t0_first_order_gt)
            second_order_gt = torch.where(mask[:, None, :, None, None], second_order_gt, x_t0_second_order_gt)
        terms = {}

        model_output = model(x_t, t, **model_kwargs) # (bsz, 8, frame, w, h)
        velocity_pred = model_output.chunk(2, dim=1)[0] # (bsz, 4, frame, w, h)
        
        second_order_input = torch.stack([velocity_pred, x_t], dim=0).mean(dim=0)
        second_order_pred = second_order_model(second_order_input, t, **model_kwargs) 
        second_order_velocity_pred = second_order_pred.chunk(2, dim=1)[0]
        
        if weights is None:
            alpha = 0.5
            scale_second_order = 0.0001
            loss_first_order = mean_flat((first_order_gt - velocity_pred).abs().pow(2), mask=mask)
            loss_second_order = scale_second_order * mean_flat((second_order_gt - second_order_velocity_pred).abs().pow(2), mask=mask)
            loss = alpha * loss_first_order + (1 - alpha) * loss_second_order
        else:
            alpha = 0.5
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss_first_order = mean_flat(weight * (first_order_gt - velocity_pred).abs().pow(2), mask=mask)
            loss_second_order = mean_flat(weight * (second_order_gt - second_order_velocity_pred).abs().pow(2), mask=mask)
            loss = alpha * loss_first_order + (1 - alpha) * loss_second_order
        terms["loss"] = loss
        terms["loss_first_order"] = loss_first_order
        terms["loss_second_order"] = loss_second_order

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise
    
    def add_noise_first_and_second_order(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        ####### Original timepoint code #######
        # timepoints = timesteps.float() / self.num_timesteps
        # timepoints = 1 - timepoints  # [1,1/1000]
        
        # # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # # expand timepoint to noise shape
        # timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        # timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])
        
        # we need to exclude 1, since 1 will make first order beta to be inf
        t = torch.rand(noise.shape[0]).to(noise.device) / (1+ 1e-6)
        t = t.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        t = t.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])
        
        a = 19.9
        b = 0.1

        # alpha_t = e^{(-1/4 a (1-t)^2-1/2 b(1-t))}
        alpha_t = torch.exp(- (1/4) * a * (1-t)**2 - (1/2) * b * (1-t))
        # first order alpha: 
        # d alpha_t / dt = alpha_t * 1/2 * (a (1-t) + b)
        first_order_alpha = alpha_t * (1/2) * (a * (1-t) + b)
        # second order alpha:
        # d^2 alpha_t / dt^2 = 1/2 * (alpha_t * (a(1-x)+b)^2 - a alpha_t)
        second_order_alpha = (1/2) * (alpha_t * (a * (1-t) + b)**2 - a * alpha_t)

        # beta_t = sqrt{1-alpha^2}
        beta_t = torch.sqrt(1 - alpha_t**2)
        # first order beta:
        # d beta_t / dt = (- alpha  / sqrt{1 - alpha^2}) * (d alpha / dt)
        first_order_beta = (- alpha_t / torch.sqrt(1 - alpha_t**2)) * first_order_alpha
        # second order beta:
        # d^2 beta_t / dt^2 = (- 1  / (1 - alpha^2) sqrt (1 - x^2)) * (d alpha / dt) + (- alpha  / sqrt{1 - alpha^2}) * (d^2 alpha / dt^2)
        second_order_beta = (- 1 / ((1 - alpha_t**2) * torch.sqrt(1 - alpha_t**2))) * first_order_alpha + first_order_beta * second_order_alpha

        x_t = alpha_t * original_samples + beta_t * noise
        first_order_gt = first_order_alpha * original_samples + first_order_beta * noise
        second_order_gt = second_order_alpha * original_samples + second_order_beta * noise
        return x_t, first_order_gt, second_order_gt

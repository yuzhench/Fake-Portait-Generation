import math
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from collections import defaultdict
from util.img_utils import clear_color
from .posterior_mean_variance import EpsilonXMeanProcessor, LearnedVarianceProcessor, LearnedRangeVarianceProcessor
# from collections import defautldict 
                  
__SAMPLER__ = {}

def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!") 
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)




def get_schedule_jump(t_T, n_sample, jump_length, jump_n_sample,
                      jump2_length=1, jump2_n_sample=1,
                      jump3_length=1, jump3_n_sample=1,
                      start_resampling=100000000):
    """
    Creates timesteps for RePaint Sampling Algorithm
    """
    jumps = {}
    for j in range(0, t_T - jump_length, jump_length):
        jumps[j] = jump_n_sample - 1

    jumps2 = {}
    for j in range(0, t_T - jump2_length, jump2_length):
        jumps2[j] = jump2_n_sample - 1

    jumps3 = {}
    for j in range(0, t_T - jump3_length, jump3_length):
        jumps3[j] = jump3_n_sample - 1

    t = t_T
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if (
            t + 1 < t_T - 1 and
            t <= start_resampling
        ):
            for _ in range(n_sample - 1):
                t = t + 1
                ts.append(t)

                if t >= 0:
                    t = t - 1
                    ts.append(t)

        if (
            jumps3.get(t, 0) > 0 and
            t <= start_resampling - jump3_length
        ):
            jumps3[t] = jumps3[t] - 1
            for _ in range(jump3_length):
                t = t + 1
                ts.append(t)

        if (
            jumps2.get(t, 0) > 0 and
            t <= start_resampling - jump2_length
        ):
            jumps2[t] = jumps2[t] - 1
            for _ in range(jump2_length):
                t = t + 1
                ts.append(t)
            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

        if (
            jumps.get(t, 0) > 0 and
            t <= start_resampling - jump_length
        ):
            jumps[t] = jumps[t] - 1
            for _ in range(jump_length):
                t = t + 1
                ts.append(t)
            jumps2 = {}
            for j in range(0, t_T - jump2_length, jump2_length):
                jumps2[j] = jump2_n_sample - 1

            jumps3 = {}
            for j in range(0, t_T - jump3_length, jump3_length):
                jumps3[j] = jump3_n_sample - 1

    ts.append(-1)

    _check_times(ts, -1, t_T)

    return ts


def _check_times(times, t_0, t_T):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= t_T, (t, t_T)

def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing=""):
    
    sampler = get_sampler(sampler)
    betas = get_named_beta_schedule(noise_schedule, steps)
         
    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised, 
                   rescale_timesteps=rescale_timesteps)

@register_sampler("ddpm")
class DDPMDiffusion:
    def __init__(self,
                 betas,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps,
                 **kwargs
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <=1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps
        
        ########### START TODO  ###########
        # Calculate the values of alpha
        # Also we will need the cumulated product of alpha.
        # And during sampling we need the value of cumulated product of alpha from
        # previous or next timestep.
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas) # cumpulated product of alphas
        self.alphas_cumprod_prev =np.concatenate(([1.0], self.alphas_cumprod[:-1]))
        self.alphas_cumprod_next = np.concatenate((self.alphas_cumprod[1:], [0.0]))
        
        ########### END TODO  ###########
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)
        
        self.mean_processor = EpsilonXMeanProcessor(betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)    
    
        self.var_processor = LearnedRangeVarianceProcessor(betas=betas)

    def p_sample_loop(self,
                      model,
                      x_start,
                      record,
                      save_root,
                      measurement=None,
                      measurement_cond_fn=None,
                      uncond=False):
        """
        The function used for sampling from noise.
        
        Args:
            model: nn.Module, the pretrained model that is used to predict the score and variance
            x_start: torch.Tensor, random noise input
            measurement: torch.Tensor, our corrupted observation
            measurement_cond_fn: conditional function used to perform conditional sampling, is None for unconditional sampling
            record: Bool, save intermediate results if True
            save_root: str, root of the directoy to save the results
            uncond: Bool, perform unconditional sampling if True, else perform conditional sampling
        """ 
        if not uncond:
            assert measurement is not None and measurement_cond_fn is not None, \
                "measurement and measurement conditional function is required for conditional sampling"
        
        img = x_start   # start from random noise
        device = x_start.device

        ############ Start TODO #############
        # Implement the sample loop
        # Call p_sample for every iteration
        # It requires only one line of code implementation here
        
        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
            
            img = self.p_sample(model,img,time)["x_t_minus_1"]
            
            img = img.detach_()
           
        ##########################################
            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))

        return img       
        
    def p_sample(self, model, x, t):
        """
        Posterior sampling process, when given the model, x_t and timestep t, it returns predicted
        x_0 and x_t_minus_1
        
        We have already provide you with the function to get the log of the variance. 
        Use self.mean_processor.get_mean_and_xstart(var_values, t), where var_values is 
        the 3:6 channels of the direct output of the model.
        example usage: log_variance = self.var_processor.get_variance(var_values, t)
        
        You can also use the helper function extract_and_expand() to extract the value 
        corresponding to timestep and expand it to the save size as the target for broadcast.
        example usage: coef1 = extract_and_expand(self.posterior_mean_coef1, t, x_start)
        
        Args:
            model: nn.Module, the UNet model, you can call model(x, t) to get the output tensor with size (B, 6, H, W)
            x: torch.Tensor, shape (1, 3, H, W), x_t
            t: torch.Tenosr, shape (1,), timestep
            
        Returns:
            output_dict: dict, contains predicted x_t_minus_1 and x_0
        """
        #####Start TODO#####
        ##### Get the predicted score and variance of the pretrained model #####
        pred_noise = model(x,t)[:,:3]
        var_values = model(x,t)[:,3:6]
        ##### End TODO #####

        log_variance = self.var_processor.get_variance(var_values, t)   # get the log  of variance
        
        #####  Start TODO   #####
        ##### get predicted x_0 and x_t_minus_1  #####
        ##### don't forget to add noise for all the steps, except for the last one  #####
        exp_log_variance = torch.exp(log_variance)
        sigma = torch.sqrt(exp_log_variance)

        coef1 = extract_and_expand(self.alphas, t, x).cuda()
        coef2 = extract_and_expand(self.alphas_cumprod, t, x).cuda()

        if t > 0:
            z = torch.randn(x.size()).cuda() #test
        else:
            z = torch.zeros_like(x)
        

        
        first_part = z * sigma
        numerator = x-((1-coef1) / torch.sqrt(1-coef2)) * pred_noise
        denominator = torch.sqrt(coef1)
        second_part = numerator / denominator
        

        x_t_minus_1 = first_part + second_part
        
        #####  End TODO   #####
        
        assert x_t_minus_1.shape == log_variance.shape == x.shape        

        output_dict = {'x_t_minus_1': x_t_minus_1}
        return output_dict



@register_sampler("ddim")
class DDIMDiffusion(DDPMDiffusion):
    
    def __init__(self, use_timesteps, **kwargs):
        
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])
    
        base_alphas_cumprod = DDPMDiffusion(**kwargs).alphas_cumprod  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        self.use_timesteps = set(use_timesteps)
        
        for i, alpha_cumprod in enumerate(base_alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)
        
    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (self.original_num_steps / self.num_timesteps)
        return t
        
    def p_sample(self, model, x, t, eta=0.0):
        ##############################################
        #####   TODO    #####
        ##### Get the predicted score and variance of the pretrained model #####
        ##### Don't forget to use _scale_timesteps to scale the timestep for calling the model prediction.
        ##### You don't need to scale the timestep for further computations of x_t_minus_1.
        ##### NOTE: Since this version of the model learns the variance along with the score function,
        ##### the output of the model would have double the number of channels as that of the input.
        ##### So assign the predicted score and variance values to the variables below.Refer to 
        ##### torch.split method.
        ##############################################
        #####  Start TODO   #####

        scaled_t = self._scale_timesteps(t)
        mode_output = model(x,scaled_t)
        pred_noise = mode_output[:, :x.shape[1]]
        ##### End TODO #####
        
        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, pred_noise)
        
        #####   TODO    #####
        ##### Step 1: Implement the variance parameter 'sigma' for DDIM sampling.   #####
        ##### Step 2: Imeplemnt x_t_minus_1 using the pred_xstart. Don't forget     #####
        #####  to add noise for all the steps, except for the t=0.                  #####
        #####                                                                       #####
        ##### You may use the function 'extract_and_expand' to expand the timestep  #####
        ##### variable 't' to the input's shape.                                    #####
        ##### Assign them to the variables x_t_minus_1.                             #####
        
        #####  Start TODO   #####
        a_t = extract_and_expand(self.alphas_cumprod,t,x).cuda()
        a_t_minus_1 = extract_and_expand(self.alphas_cumprod_prev,t,x).cuda()
        if t > 0:
            z = torch.randn_like(x).cuda()
        else:
            z = torch.zeros_like(x)
        sigma = eta*torch.sqrt(torch.sqrt((1-a_t_minus_1)/(1-a_t))*torch.sqrt(1-(a_t)/(a_t_minus_1)))

        predicted_x0 = torch.sqrt(a_t_minus_1) * ((x-torch.sqrt(1-a_t)*pred_noise)/(torch.sqrt(a_t)))
        direction_xt = (torch.sqrt(1-a_t_minus_1-(sigma**2))*pred_noise)
        x_t_minus_1 =  predicted_x0 + direction_xt

        if t > 0:
            x_t_minus_1 += sigma * z
        ##### End TODO #####
        
        return {"x_t_minus_1": x_t_minus_1, "pred_xstart": pred_xstart}
        ##############################################


    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2



@register_sampler(name='repaint')
class Repaint(DDIMDiffusion):
        
    def undo(self, image_before_step, img_after_model, est_x_0, t, debug=False):
        return self._undo(img_after_model, t)


    def _undo(self, img_out, t):
        
        beta = extract_and_expand(self.betas, t, img_out)

        img_in_est = torch.sqrt(1 - beta) * img_out + \
            torch.sqrt(beta) * torch.randn_like(img_out)

        return img_in_est
    
    
    def p_sample(
        self,
        model,
        x_t_minus_one_unknown,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        conf=None,
        pred_xstart=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x_t_minus_one_unknown: the unknown tensor at x_{t-1} (model's predicted sample in the previous timestep).
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        noise = torch.randn_like(x_t_minus_one_unknown)
        
        ##############################################
        #####   TODO    #####
        ##### Here updated sample x_t_minus_one refers to the noisy image, where the known region is    #####
        ##### obtained by adding noise to GT, and the unknown region is obtained from                   #####
        ##### x_t_minus_one_unknown (which is the predicted sample from previous timestep) and the      #####
        #####  known and unknown region are combined using the ground truth mask (gt_keep_mask).        #####
        ##### Compelete the implementation to compute the updated sample (x_t_minus_one) for the        #####
        ##### timestep t. Make use of the variables gt_keep_mask and gt, to access the                  #####
        ##### ground-truth image. and the ground-truth mask.                                            #####
        ##############################################
        
        if conf["inpa_inj_sched_prev"]:

            if pred_xstart is not None:
                
                gt_keep_mask = model_kwargs['gt_keep_mask']
                if gt_keep_mask is None:
                    gt_keep_mask = conf.get_inpa_mask(x_t_minus_one_unknown)

                gt = model_kwargs['gt']
                #####   TODO: compute x_t_minus_one     #####
                x_t_minus_one = None
            else:
                x_t_minus_one = x_t_minus_one_unknown
        
        # TODO #####
        # One-step denoising using the model: Perform a forward pass on the model.
        # Remember to scale the timestep 't' using '_scale_timesteps' method.
        # NOTE: Since this version of the model learns the variance along with the score function,
        # the output of the model would have double the number of channels as that of the input.
        # So assign the predicted score and variance values to the variables below. Refer to 
        # torch.split method.
        
        model_output = None
        pred_score, var_values = None, None
        
        
        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x_t_minus_one, t, pred_score)
        log_variance = self.var_processor.get_variance(var_values, t)   # get the log  of variance
        
        ##############################################
        ##### TODO                                                                  #####
        ##### Compute the noisy sample for timestep 't'                             #####
        ##### You should use the 'log_variance' to calculate the variance of noise  #####
        ##### to be added.                                                          #####
        ##### Assign the sample to the variable 'sample'                            #####
        ##############################################
        
        sample = None
        
        result = {"sample": sample,
                  "pred_xstart": pred_xstart, 'gt': model_kwargs.get('gt')}
        
        return result

    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised = True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=True,
        return_all=False,
        conf=None
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            conf=conf
        ):
            final = sample

        if return_all:
            return final
        else:
            return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        conf=None
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image_after_step = noise
        else:
            image_after_step = torch.randn(*shape, device=device)

        self.gt_noises = None  # reset for next image


        pred_xstart = None

        idx_wall = -1
        sample_idxs = defaultdict(lambda: 0)

        if conf["schedule_jump_params"]:
            times = get_schedule_jump(**conf["schedule_jump_params"])
            time_pairs = list(zip(times[:-1], times[1:]))
            
            if progress:
                from tqdm.auto import tqdm
                time_pairs = tqdm(time_pairs)

            for t_last, t_cur in time_pairs:
                idx_wall += 1
                t_last_t = torch.tensor([t_last] * shape[0],
                                     device=device)

                if t_cur < t_last:  # reverse
                    with torch.no_grad():
                        image_before_step = image_after_step.clone()
                        out = self.p_sample(
                            model,
                            image_after_step,
                            t_last_t,
                            clip_denoised=clip_denoised,
                            denoised_fn=denoised_fn,
                            model_kwargs=model_kwargs,
                            conf=conf,
                            pred_xstart=pred_xstart
                        )
                        image_after_step = out["sample"]
                        pred_xstart = out["pred_xstart"]

                        sample_idxs[t_cur] += 1

                        yield out

                else:
                    t_shift = conf.get('inpa_inj_time_shift', 1)

                    image_before_step = image_after_step.clone()
                    image_after_step = self.undo(
                        image_before_step, image_after_step,
                        est_x_0=out['pred_xstart'], t=t_last_t+t_shift, debug=False)
                    pred_xstart = out["pred_xstart"]

####################

@register_sampler("dps")
class DPSDiffusion:
    def __init__(self,
                 betas,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps,
                 **kwargs
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <=1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.coeff_1 = np.cumprod(alphas, axis=0)
        self.coeff_2 = np.append(1.0, self.coeff_1[:-1])
        self.coeff_3 = np.append(self.coeff_1[1:], 0.0)
        self.coeff_4 = (
            betas * np.sqrt(self.coeff_2) / (1.0 - self.coeff_1)
        )
        self.coeff_5 = (
            (1.0 - self.coeff_2)
            * np.sqrt(alphas)
            / (1.0 - self.coeff_1)
        )
        self.coeff_6 = np.sqrt(1.0 / self.coeff_1)
        self.coeff_7 = np.sqrt(1.0 / self.coeff_1 - 1)

        assert self.coeff_2.shape == (self.num_timesteps,)
        
        self.mean_processor = EpsilonXMeanProcessor(betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)    
    
        self.var_processor = LearnedRangeVarianceProcessor(betas=betas)

        print('DPS Initialized!')

    def p_sample_loop(self,
                      model,
                      x_start,
                      record,
                      save_root,
                      measurement=None,
                      measurement_cond_fn=None,
                      uncond=False):
        """
        The function used for sampling from noise.
        
        Args:
            model: nn.Module, the pretrained model that is used to predict the score and variance
            x_start: torch.Tensor, random noise input
            measurement: torch.Tensor, our corrupted observation
            measurement_cond_fn: conditional function used to perform conditional sampling, is None for unconditional sampling
            record: Bool, save intermediate results if True
            save_root: str, root of the directoy to save the results
            uncond: Bool, perform unconditional sampling if True, else perform conditional sampling
        """ 
        if not uncond:
            assert measurement is not None and measurement_cond_fn is not None, \
                "measurement and measurement conditional function is required for conditional sampling"
        
        img = x_start   # start from random noise
        device = x_start.device
        
        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
              
            img = img.requires_grad_()  # x_i
            out = self.p_sample(x=img, t=time, model=model)
            img = measurement_cond_fn(x_t_minus_one=out['x_t_minus_1'],
                                    measurement=measurement,
                                    noisy_measurement=None,
                                    x_i=img,
                                    x_0_hat=out['pred_x_0'])
            
            img = img.detach_()
           
            if record:
                if idx % 10 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))

        return img       
        
    def p_sample(self, model, x, t):
        """
        Posterior sampling process, when given the model, x_t and timestep t, it returns predicted
        x_0 and x_t_minus_1

        Args:
            model: nn.Module, the UNet model, you can call model(x, t) to get the output tensor with size (B, 6, H, W)
            x: torch.Tensor, shape (1, 3, H, W), x_t
            t: torch.Tenosr, shape (1,), timestep
            
        Returns:
            output_dict: dict, contains predicted x_t_minus_1 and x_0
        """
        pred_score = None
        var_values = None
        
        model_output = model(x, t)
        pred_score, var_values = torch.split(model_output, x.shape[1], dim=1)   # (1, 6, H, W)

        log_variance = self.var_processor.get_variance(var_values, t)   # get the log  of variance
        
        pred_x_0 = None
        x_t_minus_1 = None
        
        coef1_x0 = extract_and_expand(self.coeff_6, t, x)
        coef2_x0 = extract_and_expand(self.coeff_7, t, pred_score)
        
        pred_x_0 = (coef1_x0 * x - coef2_x0 * pred_score).clip(-1, 1)
        
        coef1_x_t_minus_1 = extract_and_expand(self.coeff_4, t, pred_x_0)
        coef2_x_t_minus_1 = extract_and_expand(self.coeff_5, t, pred_score)
        
        x_t_minus_1 = coef1_x_t_minus_1 * pred_x_0 + coef2_x_t_minus_1 * x
        
        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            x_t_minus_1 += torch.exp(0.5 * log_variance) * noise

        assert x_t_minus_1.shape == log_variance.shape == pred_x_0.shape == x.shape        

        output_dict = {'x_t_minus_1': x_t_minus_1, 'pred_x_0': pred_x_0}
        return output_dict


# =================
# Helper functions
# =================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_min=0.0001, beta_max=0.02):
    """
    Get a pre-defined beta schedule for the given name.
    
    Args:
        schedule_name: str, name of the variance schedule, 'linear' or 'cosine'
        num_diffusion_timesteps: int, number of the entire diffusion timesteps
        beta_min: float, minimum value of beta
        beta_max: float, maximum value of beta
        
    Returns:
        betas: np.ndarray, a 1-d array of size num_diffusion_timesteps, contains all the beta for each timestep

    """
    betas = [0]
    if schedule_name == "linear":
        ########## START TODO ##########
        # Implement the linear schedule
        # Uniformly divide the [beta_min, beta_max) to num_diffusion_timesteps values.

        betas = np.arange(beta_min, beta_max, (beta_max - beta_min) / num_diffusion_timesteps)
        ########## END TODO ##########
    elif schedule_name == "cosine":
        ########## START TODO ##########
        # Implement the cosine schedule
        # Assume s = 0.008 and beta_clip=0.999
        s = 0.008
        beta_clip = 0.999

        # betas = np.zeros(num_diffusion_timesteps)
        old_alpha = 1
        normalize_factor = math.cos((s/(1+s))*(math.pi/2))**2
        T = num_diffusion_timesteps
        for t in range(1,T):
            alpha = math.cos((((t/T)+s)/(1+s))*(math.pi/2))**2 / normalize_factor
            min_beta = min(beta_clip,1-(alpha/old_alpha))
            betas.append(min_beta)
            #update the new alpha
            old_alpha = alpha
        del betas[0]
        betas.append(0.999)

        ########## End TODO ##########
         

        # Add a value to the end of betas
         
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
    
    return betas


# ================
# Helper function
# ================


def extract_and_expand(array, time, target):
    """
    array: A numpy array of shape (T,), where T is the number of timesteps of diffusion.
    time: An integer which denotes the time you want to extract.
    target: A torch tensor, the shape of which will be used to determine the shape of the output.
    
    This function extracts the value at a timestep 'time' and expands them to an array of shape 'target.shape'
    Example: 
        Consider extracting the array self.betas at timestep 20.
        array = self.betas -> Numpy array of shape (1000,)
        time = 20
        target = Assume a torch tensor of shape (5, 3, 256, 256)
        
        Calling the function extract_and_expand as below
        
        'extract_and_expand(self.betas, 20, target)'
        
        returns the value 'self.betas[20]' expanded to the shape (5, 3, 256, 256).
        
        You may use this function to extract the value at timestep 't' for tensors like alphas, alphas_cumprod,
        and reshape the value to the shape of target.
    """
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)
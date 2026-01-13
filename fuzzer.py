import time
import torch
import sys
import matplotlib.pyplot as plt
import signal
import numpy as np
from model_wrappers import ModelWrapper
from coverage_metrics import Coverage
import json
import torch.nn as nn
import os
import random
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms

to_pil = transforms.ToPILImage()


class Fuzzer:
    def __init__(self, seeds_loader: DataLoader, model_wrapper: ModelWrapper, coverage: Coverage, target_label=None, epochs=10000, timeout=6*60*60, ae_dir="adversarial-examples/default", coverage_guided=True, random_mutation=False, noise_range=(-8, 8)):
        
        self.model_wrapper = model_wrapper
        self.coverage = coverage
        self.seeds = list(seeds_loader)
        self.seeds_loader = seeds_loader
        self.print_every = 5
        self.epochs = epochs
        self.timeout = timeout
        self.ae_dir = ae_dir
        self.coverage_guided = coverage_guided
        self.target_label = target_label
        self.random_mutation = random_mutation
        self.noise_range = noise_range

        self.delta_time = 0
        self.coverage_gains = [coverage.current.item() if isinstance(coverage.current, torch.Tensor) else coverage.current]
        self.delta_times = [0]
        self.overall_ae_counts = [0]

        low, high = self.noise_range
        self.noises = [
            torch.rand(self.seeds[i][0][0].shape) * (high - low) + low
            for i in range(len(self.seeds))
        ]

        self.ae_map = torch.zeros((len(self.seeds_loader), self.seeds_loader.batch_size), dtype=torch.long)

        self.noise_scores = [1]*len(self.noises)
        self.losses = [None]*len(self.noises)

        self.mean_pred_loss = 0      # running average
        self.mean_gain = 0           # running average
        self.mean_nat_loss =0       # running average

        signal.signal(signal.SIGINT, self.exit)

    def exit(self, sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)

    def can_terminate(self):
        if self.epochs != -1 and self.epoch > self.epochs:
            print(f"Reached max epochs: {self.epochs}")
            return True
        if self.timeout != -1 and self.delta_time > self.timeout:
            print(f"Reached timeout: {self.timeout} seconds")
            return True
        return False

    def compute_coverage_gain(self, preprocessed_inputs):
        cov_dict = self.coverage.calculate(preprocessed_inputs)
        gain = self.coverage.gain(cov_dict)
        if gain is not None:
            _, gain_amt = self.calculate_coverage_gain(gain)
            self.coverage.update(cov_dict, gain)
        else:
            gain_amt = torch.tensor(0.0001, requires_grad=True)
        return gain_amt


    def mutate(self, noise, inputs, ground_labels):
        noise = noise.clone().detach().requires_grad_(True)

        preprocessed_inputs = self.model_wrapper.preprocess(inputs + noise)
        gain = self.compute_coverage_gain(preprocessed_inputs)
        prediction_loss = self.model_wrapper.compute_loss(
            preprocessed_inputs, ground_labels, self.target_label
        )
        naturalness_loss = self.model_wrapper.naturalness_loss(inputs, noise)

        # ---- update running averages for scaling ----
        alpha = 0.9
        self.mean_pred_loss = (1 - alpha) * self.mean_pred_loss + alpha * abs(prediction_loss.item())
        self.mean_nat_loss  = (1 - alpha) * self.mean_nat_loss  + alpha * naturalness_loss.item()
        self.mean_gain      = (1 - alpha) * self.mean_gain      + alpha * gain.item()

        scale_pred = 1.0 / (self.mean_pred_loss + 1e-6)
        scale_gain = 1.0 / (self.mean_gain + 1e-6)
        scale_nat  = 1.0 / (self.mean_nat_loss + 1e-6)

        # composite loss
        loss = -scale_gain * gain + scale_pred * prediction_loss + scale_nat * naturalness_loss
        
        # ---- update ----
        if loss.requires_grad:
            loss.backward()
            with torch.no_grad():
                if self.random_mutation:
                    # random direction (unguided mutation)
                    step = torch.randn_like(noise)
                    step = step / (step.norm(p=2) + 1e-6)
                else:
                    # gradient-guided direction
                    grad = noise.grad
                    grad_norm = grad.norm(p=2) + 1e-6
                    step = grad / grad_norm            # unit vector direction
                lr = 0.1   
                noise = noise - lr * step

                noise = self.model_wrapper.clamp_noise(noise, inputs)

        return noise.clone().detach(), loss.item()


    def calculate_coverage_gain(self, gain):
        """Returns if there was coverage gain or not and the gain amount"""
        if gain is not None:
            if isinstance(gain, tuple):
                return gain[0] > 0, gain[0]
            else:
                return gain > 0, gain
        else:
            return False, torch.tensor(0.0)

    def is_adversarial(self, mutated_noise, seed_idx):
        inputs, ground_truths, *_ = self.seeds[seed_idx]
        original_predictions = self.model_wrapper.predict_outputs(self.model_wrapper.preprocess(inputs))
        mutated_inputs = inputs + mutated_noise
        noisy_inputs = self.model_wrapper.clamp(mutated_inputs)

        total = 0
        miss_classification = 0

        adversarial_inputs = []
        adversarial_input_labels = []

        adversarial_original_inputs = []
        adversarial_original_input_labels = []

        with torch.no_grad():
            preprocessed_data = self.model_wrapper.preprocess(noisy_inputs)

            predicted = self.model_wrapper.predict_outputs(preprocessed_data)
            total += noisy_inputs.size(0)
            
            adversarial_mask = self.model_wrapper.adversarial_mask(predicted, original_predictions, ground_truths, target_label=self.target_label)

            current_bs = adversarial_mask.size(0)
            self.ae_map[seed_idx, :current_bs] += adversarial_mask.to(torch.long)

            adversarial_inputs.extend(torch.unbind(noisy_inputs[adversarial_mask.to(noisy_inputs.device)], dim=0))
            adversarial_input_labels.extend([pred for pred, keep in zip(predicted, adversarial_mask) if keep])

            adversarial_original_inputs.extend(torch.unbind(inputs[adversarial_mask.to(inputs.device)], dim=0))

            adversarial_original_input_labels.extend([(ground_truth, original_prediction) for ground_truth, original_prediction, keep in zip(ground_truths, original_predictions, adversarial_mask) if keep])

            miss_classification += adversarial_mask.sum().item()

        mis_rate = 100 * miss_classification / total
        is_adversarial = miss_classification>0 

        if is_adversarial:
            print(f"Adversarial: {miss_classification} ({mis_rate:.2f}%)")
        return is_adversarial, list(zip(adversarial_inputs, adversarial_input_labels, adversarial_original_inputs, adversarial_original_input_labels))

    def save_adversarial_examples(self, adversarial_inputs):
        i=1
        for ae, ae_label, orig, (ground_truth, original_prediction) in adversarial_inputs:
            self.model_wrapper.save_adversarial_example(f"{self.epoch}_{i}", orig, ae, ae_label, original_prediction, ground_truth, self.target_label, self.ae_dir)
            i += 1

    def select_next(self):
        if random.random() < 0.05:
            noise_index = random.randint(0, len(self.noises) - 1)
        else:
            total = sum(self.noise_scores)
            probs = [score / total for score in self.noise_scores]
            noise_index = random.choices(range(len(self.noises)), weights=probs, k=1)[0]
    
        return self.noises[noise_index], noise_index

    def run(self):
        print("Started..")
        self.epoch = 0
        overall_ae_count = 0
        start_time = time.time()

        mean_ae = 0
        mean_loss = 0
        while not self.can_terminate():

            noise, seed_idx = self.select_next() 
            inputs, ground_labels, *_ = self.seeds[seed_idx]
            mutated_noise, loss = self.mutate(noise,inputs, ground_labels)

            self.noises[seed_idx] = mutated_noise

            
            is_adversarial, adversarial_inputs = self.is_adversarial(mutated_noise, seed_idx)
            self.coverage_gains.append(self.coverage.current.item() if isinstance(self.coverage.current, torch.Tensor) else self.coverage.current)
            ae_count = 0
            if is_adversarial:
    
                ae_count = len(adversarial_inputs)
                overall_ae_count += ae_count

                self.save_adversarial_examples(adversarial_inputs)


            # ---- Compute recent improvements ----
            if self.losses[seed_idx] is None:
                delta_loss = 0
            else:
                delta_loss = self.losses[seed_idx] - loss           # Loss improvement (positive is good)
            delta_ae   = ae_count                  # New AEs found this iteration (already "recent")

            # ---- Update running means (recent weighted more) ----
            mean_loss = 0.1 * mean_loss + 0.9 * max(delta_loss, 0)
            mean_ae   = 0.1 * mean_ae   + 0.9 * max(delta_ae, 0)

            # ---- Normalize (so scale stays balanced) ----
            norm_loss = max(delta_loss, 0) / (mean_loss + 1e-6)
            norm_ae   = max(delta_ae, 0)  / (mean_ae   + 1e-6)

            # ---- Priority update (positive if improved) ----
            priority_update = norm_loss + norm_ae   # Both ∈ [0, ∞)

            # ---- Boost for improvements, decay otherwise ----
            if priority_update > 0:
                self.noise_scores[seed_idx] *= (1 + priority_update)
            else:
                self.noise_scores[seed_idx] *= 0.95

            # ---- Prevent scores from collapsing ----
            self.noise_scores[seed_idx] = max(self.noise_scores[seed_idx], 1e-6)
            # ---- Cap scores to prevent extreme outliers ----
            mean_score = sum(self.noise_scores) / len(self.noise_scores)
            cap = 2.0 * mean_score
            self.noise_scores[seed_idx] = min(self.noise_scores[seed_idx], cap)
            self.losses[seed_idx] = loss
      

            self.epoch += 1
            self.delta_time = time.time() - start_time
            if self.epoch%self.print_every == 0:
                print(f"#Epochs: {self.epoch}")
                print(f"Time: {self.delta_time:.2f} seconds")
                print(f"Coverage: {self.coverage.current}")
                print(f"# AE: {overall_ae_count}")

            self.delta_times.append(self.delta_time)
            self.overall_ae_counts.append(overall_ae_count)

        print(f"Coverage: {self.coverage.current}")
        print(f"# AE: {overall_ae_count}")
        print(f"Duration: {time.time() - start_time}")


        for i in range(len(self.noises)):
            self.model_wrapper.save_poison(i, self.noises[i], self.ae_dir)
            

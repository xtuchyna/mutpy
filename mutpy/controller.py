import random
import sys
import time
from copy import deepcopy
import re

# for doc comparisson
import spacy
import en_core_web_sm
import pandas as pd

import random
import os


from mutpy import views, utils

def get_full_test_name(test_name) -> tuple:
    regex = r'(\S+)\s+\((\S+)\)'
    match = re.match(regex, test_name)
    return match.group(2)

def get_rtf_series(first_killers, rapfd_constraint_m):
    """Get RTF series give test constraint m."""
    return map(lambda x: x if rapfd_constraint_m >= x else 0, first_killers)

def get_first_killers(kill_order_per_mutations):
    """From list of killers for all killed mutants, return the first killer for each mutant."""
    return [test_orders[0] for test_orders in kill_order_per_mutations]

def get_cumulative_step_sum_of_covered_fault_area(rtf_series, all_mutants):
    """Get cumulative step sum of covered fault area."""
    faults_detected_by_first_m = map(lambda phi: phi != 0, rtf_series)
    return len( list( faults_detected_by_first_m ) ) / all_mutants

def write_into_file(data: pd.DataFrame, file_full_path: str) -> None:
    """Writes/appends into aggregation csv EDA file."""
    file_exists = os.path.isfile(file_full_path)
    data.to_csv(file_full_path, mode='a', header=not file_exists, index=False)


class CompareOutputs:

    def __init__(self, theta):
        self.nlp = spacy.load("en_core_web_sm")
        self.theta = theta

    def is_same_output(self, text1, text2):
        # Normalize the outputs by removing leading/trailing whitespace and newlines
        normalized_text1 = re.sub(r'\s+', ' ', text1.strip())
        normalized_text2 = re.sub(r'\s+', ' ', text2.strip())

        if normalized_text1 == normalized_text2:
            return True
        
        return self.is_same_using_cosine(normalized_text1, normalized_text2)


    # def is_same_using_spaceless(text1, text2):
    #     # Remove all spaces, including newlines, tabs, and regular spaces
    #     cleaned_text1 = re.sub(r'\s+', '', text1)  # \s+ matches any whitespace character (space, newline, tab) one or more times
    #     cleanmed_text2 = re.sub(r'\s+', '', text2)
    #     return cleaned_text1 == cleanmed_text2

    def is_same_using_cosine(self, output_original, output_mutant):
        doc1 = nlp(output_original)
        doc2 = nlp(output_mutant)
        
        # Compute cosine similarity between document vectors
        similarity = doc1.similarity(doc2)
        return similarity > self.theta


nlp = en_core_web_sm.load()


def is_same_output(output1, output2) -> bool:
    # Normalize the outputs by removing leading/trailing whitespace and newlines
    normalized_output1 = re.sub(r'\s+', ' ', output1.strip())
    normalized_output2 = re.sub(r'\s+', ' ', output2.strip())
    
    # Compare the normalized outputs
    return normalized_output1 == normalized_output2


class TestsFailAtOriginal(Exception):

    def __init__(self, result=None):
        self.result = result


class MutationScore:

    def __init__(
            self,
            result,
            rapfd_constraint_m,    
        ):

        self.killer_matrix = {
            get_full_test_name(test.name):[] for test in result.passed + result.failed
        }
        self.test_size = len(self.killer_matrix.keys())
        self.test_order = result.test_order

        self.killed_mutants = 0
        self.timeout_mutants = 0
        self.incompetent_mutants = 0
        self.survived_mutants = 0
        self.covered_nodes = 0
        self.all_nodes = 0

        self.overall_mutations = []
        self.per_mutant_stats = {}

        self.apfd_buffer = []
        self.kill_order_per_mutations = []

        self.apfd = None
        self.rapfd = None

        self.rapfd_constraint_m = rapfd_constraint_m

        self.rtf_series = None

    def get_apfd_score(self):
        if self.apfd is not None:
            return self.apfd
        
        self.apfd = 1 - sum(
            [killer_tests[0] for killer_tests in self.kill_order_per_mutations]
        ) / (self.test_size * self.all_mutants) + 1 / (2 * self.test_size)

        return self.apfd
    
    def get_rtf_series(self):
        if self.rtf_series is not None:
            return self.rtf_series

        first_killers = [test_orders[0] for test_orders in self.kill_order_per_mutations]
        return map(lambda x: x if self.rapfd_constraint_m >= x else 0, first_killers)

    def get_cumulative_step_sum_of_covered_fault_area(self):
        # faults_detected_by_first_m = map(lambda phi: phi != 0, self.get_rtf_series())
        faults_detected_by_first_m = [observed_fault for observed_fault in self.get_rtf_series() if observed_fault != 0]
        return len( list( faults_detected_by_first_m ) ) / self.all_mutants


    def get_rapfd_score(self):
        if self.rapfd is not None:
            return self.rapfd
        
        p_m = self.get_cumulative_step_sum_of_covered_fault_area()

        self.rapfd = p_m - sum( self.get_rtf_series() ) / (self.rapfd_constraint_m * self.all_mutants)
        return self.rapfd
        
    def get_random_rapfd_score(self):

        # Original list of indices
        original_indices = [i for i in range(1,self.test_size+1)]

        # Shuffle to get a new order
        shuffled = original_indices.copy()
        random.shuffle(shuffled)

        # Create mapping from original index → new index
        swap_map = {original: new for new, original in enumerate(shuffled, start=1)}

        # Accessing the new swapped index
        def get_swapped_index(original_index):
            return swap_map[original_index]

        # remap test order with swap map
        new_test_order = {k:get_swapped_index(v) for k,v in self.test_order.items()}

        # print("Original test order: ", self.test_order)
        # print("New test order: ", new_test_order)

        # remap killers order with swap map and sort (to logically retain killer order by sorting)
        # new_killers_orders = [
        #     sorted( swap_map[ith_test] for test in self.kill_order_per_mutations for ith_test in test )
        # ]


        new_killers_orders = []
        for killer_order in self.kill_order_per_mutations:
            new_killer_order_per_killed_mutant = sorted( [swap_map[ith_test] for ith_test in killer_order] )
            new_killers_orders.append(new_killer_order_per_killed_mutant)


        # print("Original killers orders: ", self.kill_order_per_mutations)
        # print("New killers orders: ", new_killers_orders)

        first_killers = get_first_killers(new_killers_orders)
        rtf_series = get_rtf_series(first_killers, self.rapfd_constraint_m)

        # compute rapfd as previously
        p_m = get_cumulative_step_sum_of_covered_fault_area(rtf_series, self.all_mutants)
        return p_m - sum( rtf_series ) / (self.rapfd_constraint_m * self.all_mutants)

    def count(self):
        bottom = self.all_mutants - self.incompetent_mutants
        return (((self.killed_mutants + self.timeout_mutants) / bottom) * 100) if bottom else 0

    def inc_killed(self):
        self.killed_mutants += 1

    def inc_timeout(self):
        self.timeout_mutants += 1

    def inc_incompetent(self):
        self.incompetent_mutants += 1

    def inc_survived(self):
        self.survived_mutants += 1

    def update_coverage(self, covered_nodes, all_nodes):
        self.covered_nodes += covered_nodes
        self.all_nodes += all_nodes

    @property
    def all_mutants(self):
        return self.killed_mutants + self.timeout_mutants + self.incompetent_mutants + self.survived_mutants


class MutationController(views.ViewNotifier):

    def __init__(self, runner_cls, target_loader, test_loader, views, mutant_generator,
                 timeout_factor=5, disable_stdout=False, mutate_covered=False, mutation_number=None,
                 theta_factor=0.8, rapfd_constraint_m=5):
        super().__init__(views)
        self.target_loader = target_loader
        self.test_loader = test_loader
        self.mutant_generator = mutant_generator
        self.timeout_factor = timeout_factor
        self.stdout_manager = utils.StdoutManager(disable_stdout)
        self.mutation_number = mutation_number
        self.runner = runner_cls(self.test_loader, self.timeout_factor, self.stdout_manager, mutate_covered)

        self.test_results_matrix = {}
        self.original_failed_tests = {}

        self.apfd_buffer = []
        self.comparator = CompareOutputs(theta=theta_factor)

        self.rapfd_constraint_m = rapfd_constraint_m

    def save_per_test(self, folder: str) -> None:
        csv_scores = []
        for test_name, killed_operators in self.score.killer_matrix.items():
            kill_count = len(killed_operators)
            per_test_score = kill_count / self.score.all_mutants
            csv_scores.append([test_name, per_test_score])

        data = pd.DataFrame(csv_scores, columns=["test_name", "per_test_score"])
        write_into_file(data, folder + '/per_test.csv')

    def initialize_per_mutant_entry(op, post_process_dict) -> None:
        if op not in post_process_dict:
            for stat_field in ["killed", "survived", "per_test_kills", "per_test_survives"]:
                post_process_dict[op][stat_field] = 0

    # def save_per_mutant(self, folder: str) -> None:
    #     csv_scores = []

    #     postprocessed_mutants_score = {}
    #     for test, mut_operators in self.score.killer_matrix.items():
    #         for op in mut_operators:
    #             initialize_per_mutant_entry(op, postprocessed_mutants_score)
    #             postprocessed_mutants_score[op]["killed"] = 0
    #             postprocessed_mutants_score[op]["survived"]

    #     for mutant, stats in self.score.per_mutant_stats.items():
    #         # per_mutant_score = num_of_killed / self.score.all_mutants * 100
    #         killed = stats["killed"]
    #         overall = stats["generated"]
    #         survived = stats["survived"]

    #         csv_scores.append({
    #             "test_module": self.test_module_name,
    #             "mutant_type": mutant,
    #             "generated": overall,
    #             "killed": killed,
    #             "survived": survived,
    #         })

    #     pd.DataFrame(csv_scores).to_csv(folder + '/per_mutant.csv', index=False)

    def get_target_file_name(self):
        full_path = self.target_loader.names[0]
        return os.path.basename( full_path )

    def save_per_suite(self, folder: str) -> None:
        csv_score = {
            "test_module_name" : self.test_module_name,
            "target_file": self.get_target_file_name(),
            "mutation_score": self.score.count(),
            "time_elapsed": self.duration,
            "all_mutants": self.score.all_mutants,
            "killed": self.score.killed_mutants,
            "killed_prctg": self.score.killed_mutants / self.score.all_mutants * 100,
            "survived": self.score.survived_mutants,
            "survived_prctg": self.score.survived_mutants / self.score.all_mutants * 100,
            "incopetent" : self.score.incompetent_mutants,
            "incopetent_prctg": self.score.incompetent_mutants / self.score.all_mutants * 100,
            "timeout": self.score.timeout_mutants,
            "timeout_prctg": self.score.timeout_mutants / self.score.all_mutants * 100,
            "rapfd_score": self.score.get_rapfd_score(),
            "random_rapfd_score": self.score.get_random_rapfd_score(),
        }

        data = pd.DataFrame([csv_score])
        write_into_file(data, folder + '/per_suite.csv')


    def save_eda(self, folder):
        # Save to CSV per_test.csv
        # testsuite_name | test_name | per_test score
        self.save_per_test(folder)

        # Save to CSV per_mutant.csv
        # testsuite_name | mutants in str | per_mutant score
        # self.save_per_mutant(folder)

        # Save to CSV per_suite.csv
        # testsuite_name | mut score | mt time | all | killed | killed_% | survived | surv_% | incompetent | incmpt_% | timeout | timeout_%
        self.save_per_suite(folder)



    def run(self):
        self.notify_initialize(self.target_loader.names, self.test_loader.names)
        try:
            timer = utils.Timer()
            self.run_mutation_process()
            self.duration = timer.stop()
            self.notify_end(self.score, self.duration)
        except TestsFailAtOriginal as error:
            self.notify_original_tests_fail(error.result)
            sys.exit(-1)
        except utils.ModulesLoaderException as error:
            self.notify_cant_load(error.name, error.exception)
            sys.exit(-2)

    def run_mutation_process(self):
        try:
            # test_modules, total_duration, number_of_tests = self.load_and_check_tests()
            test_modules, total_duration, number_of_tests = self.load_and_check_tests()

            results = [module[1] for module in test_modules]
            passed = [test for res in results for test in res.passed]
            failed = [test for res in results for test in res.failed]

            for test in failed:
                test_name = get_full_test_name(test.name)
                self.original_failed_tests[test_name] = test

            self.print_test_results(passed, failed)
            
            # Todo comment out for experimental failing version
            self.notify_passed(test_modules, number_of_tests)

            self.notify_start()

            # self.score = MutationScore(
            #     test_size = len(passed) + len(failed),
            #     test_order = self.test_order,
            #     rapfd_constraint_m = self.rapfd_constraint_m
            # )

            # test module tuple not used, only first element module because of *_
            for target_module, to_mutate in self.target_loader.load([module for module, *_ in test_modules]):
                self.mutate_module(target_module, to_mutate, total_duration)
        except KeyboardInterrupt:
            pass

    def load_and_check_tests(self):
        test_modules = []
        number_of_tests = 0
        total_duration = 0
        for test_module, target_test in self.test_loader.load():
            result, duration = self.run_test(test_module, target_test)
            # if result.was_successful():
            # Allow failures
            # sort tests by their name, both fails and passes
            # test_execution_order = sorted(result.passed + result.failed, key=lambda x: x.name)
            test_modules.append((test_module, result, target_test, duration))
            self.test_module_name = test_module.__name__
            # TODO provide result.was_success to the tuple list?
            # test_modules.append((test_module, result.was_successful(), target_test, duration))
            # else:
            #     raise TestsFailAtOriginal(result)
            
            self.score = MutationScore(
                result,
                rapfd_constraint_m = self.rapfd_constraint_m
            )

            # TODO support multiple test suites with multiple test orders
            self.test_order = result.test_order

            number_of_tests += result.tests_run()
            total_duration += duration

        return test_modules, total_duration, number_of_tests

    def run_test(self, test_module, target_test):
        return self.runner.run_test(test_module, target_test)

    @utils.TimeRegister
    def mutate_module(self, target_module, to_mutate, total_duration):
        target_ast = self.create_target_ast(target_module)
        coverage_injector, coverage_result = self.inject_coverage(target_ast, target_module)
        if coverage_injector:
            self.score.update_coverage(*coverage_injector.get_result())
        for mutations, mutant_ast in self.mutant_generator.mutate(target_ast, to_mutate, coverage_injector,
                                                                  module=target_module):
            mutation_number = self.score.all_mutants + 1
            if self.mutation_number and self.mutation_number != mutation_number:
                self.score.inc_incompetent()
                continue
            self.notify_mutation(mutation_number, mutations, target_module, mutant_ast)
            mutant_module = self.create_mutant_module(target_module, mutant_ast)
            if mutant_module:
                self.run_tests_with_mutant(total_duration, mutant_module, mutations, coverage_result)
            else:
                self.score.inc_incompetent()

    def inject_coverage(self, target_ast, target_module):
        return self.runner.inject_coverage(target_ast, target_module)

    @utils.TimeRegister
    def create_target_ast(self, target_module):
        with open(target_module.__file__) as target_file:
            return utils.create_ast(target_file.read())

    @utils.TimeRegister
    def create_mutant_module(self, target_module, mutant_ast):
        try:
            with self.stdout_manager:
                return utils.create_module(
                    ast_node=mutant_ast,
                    module_name=target_module.__name__
                )
        except BaseException as exception:
            self.notify_incompetent(0, exception, tests_run=0)
            return None


    def update_apfd_list(self, real_killers, mutations):
        # survived mutants are not included in APFD, it applies only for *REVEALED* faults
        if real_killers is None or real_killers == []:
            return
    
        # failed tests are returned in order of execution
        # TODO make this nicer and less corporate
        mut_killer_test_names = [get_full_test_name(test.name) for test in real_killers]

        # TODO make it dictionary
        self.score.kill_order_per_mutations.append( [self.score.test_order[t] for t in mut_killer_test_names] )

    def update_per_test_matrix(self, result, mutations):
        # of FOM, only one operator in mutations
        # If HOM, there will be multiple mutations.
        mutated_operators = [mutation.operator.__name__ for mutation in mutations]

        if result is None:
            # if no result, mutant survived
            return
        
        real_killers = []
        for killer in result.killer:
            full_test_name = get_full_test_name(killer.name)

            # check if subtest already failed before originally
            if full_test_name in self.original_failed_tests:

                # if yes, compare two outputs
                before_mutation_out = self.original_failed_tests[full_test_name].long_message
                after_mutation_out = killer.long_message 

                if self.comparator.is_same_output(before_mutation_out, after_mutation_out):
                    # if same, mutation did not change the testing behaviour
                    continue


            if full_test_name not in self.score.killer_matrix:
                self.score.killer_matrix[full_test_name] = []

            # update per test metrics
            self.score.killer_matrix[full_test_name].append(mutated_operators)
            real_killers.append(killer)
        
        return real_killers


    def update_mutant_stats(self, real_killers, mutations):
        mutated_operators = [mutation.operator.__name__ for mutation in mutations]

        self.score.overall_mutations.append(mutated_operators)

        # if HOM, tuple is used as ID
        mutant_id = str(mutated_operators)
        if mutant_id not in self.score.per_mutant_stats:
            self.score.per_mutant_stats[mutant_id] = {
                "generated": 0,
                "killed": 0,
                "survived": 0,
            }

        # update generated count
        killed_muts = len(real_killers)
        self.score.per_mutant_stats[mutant_id]["generated"] += 1
        self.score.per_mutant_stats[mutant_id]["killed_per_test"] += killed_muts
        self.score.per_mutant_stats[mutant_id]["survived"] += self.score.test_size - killed_muts


    def run_tests_with_mutant(self, total_duration, mutant_module, mutations, coverage_result):
        result, duration = self.runner.run_tests_with_mutant(total_duration, mutant_module, mutations, coverage_result)

        # removes original test failures if they are not killers
        real_killers = self.update_per_test_matrix(result, mutations)

        # self.update_mutant_stats(real_killers, mutations)
        self.update_apfd_list(real_killers, mutations)

        # make mutation score compatible with per test metrics
        self.update_score_and_notify_views(result, duration, real_killers)

    def update_score_and_notify_views(self, result, mutant_duration, real_killers):
        # due to internal mutpy logic, we need to leave timeouts for entire suite
        if not result:
            self.update_timeout_mutant(mutant_duration)

        # iterate per test results.killed
        # if mutant was killed, update per test matrix with killers BUT CHECK FOR ORIGINAL FAILURES

        elif result.is_incompetent:
            self.update_incompetent_mutant(result, mutant_duration)
        elif len(real_killers) == 0:
            self.update_survived_mutant(result, mutant_duration)
        else:
            self.update_killed_mutant(result, mutant_duration, real_killers)

    def update_timeout_mutant(self, duration):
        self.notify_timeout(duration)
        self.score.inc_timeout()

    def update_incompetent_mutant(self, result, duration):
        self.notify_incompetent(duration, result.exception, result.tests_run)
        self.score.inc_incompetent()

    def update_survived_mutant(self, result, duration):
        self.notify_survived(duration, result.tests_run)
        self.score.inc_survived()

    def update_killed_mutant(self, result, duration, real_killers):
        # use test names in list
        self.notify_killed(duration, str([kill.name for kill in real_killers]), result.exception_traceback, result.tests_run)
        self.score.inc_killed()


class HOMStrategy:

    def __init__(self, order=2):
        self.order = order

    def remove_bad_mutations(self, mutations_to_apply, available_mutations, allow_same_operators=True):
        for mutation_to_apply in mutations_to_apply:
            for available_mutation in available_mutations[:]:
                if mutation_to_apply.node == available_mutation.node or \
                        mutation_to_apply.node in available_mutation.node.children or \
                        available_mutation.node in mutation_to_apply.node.children or \
                        (not allow_same_operators and mutation_to_apply.operator == available_mutation.operator):
                    available_mutations.remove(available_mutation)


class FirstToLastHOMStrategy(HOMStrategy):
    name = 'FIRST_TO_LAST'

    def generate(self, mutations):
        mutations = mutations[:]
        while mutations:
            mutations_to_apply = []
            index = 0
            available_mutations = mutations[:]
            while len(mutations_to_apply) < self.order and available_mutations:
                try:
                    mutation = available_mutations.pop(index)
                    mutations_to_apply.append(mutation)
                    mutations.remove(mutation)
                    index = 0 if index == -1 else -1
                except IndexError:
                    break
                self.remove_bad_mutations(mutations_to_apply, available_mutations)
            yield mutations_to_apply


class EachChoiceHOMStrategy(HOMStrategy):
    name = 'EACH_CHOICE'

    def generate(self, mutations):
        mutations = mutations[:]
        while mutations:
            mutations_to_apply = []
            available_mutations = mutations[:]
            while len(mutations_to_apply) < self.order and available_mutations:
                try:
                    mutation = available_mutations.pop(0)
                    mutations_to_apply.append(mutation)
                    mutations.remove(mutation)
                except IndexError:
                    break
                self.remove_bad_mutations(mutations_to_apply, available_mutations)
            yield mutations_to_apply


class BetweenOperatorsHOMStrategy(HOMStrategy):
    name = 'BETWEEN_OPERATORS'

    def generate(self, mutations):
        usage = {mutation: 0 for mutation in mutations}
        not_used = mutations[:]
        while not_used:
            mutations_to_apply = []
            available_mutations = mutations[:]
            available_mutations.sort(key=lambda x: usage[x])
            while len(mutations_to_apply) < self.order and available_mutations:
                mutation = available_mutations.pop(0)
                mutations_to_apply.append(mutation)
                if not usage[mutation]:
                    not_used.remove(mutation)
                usage[mutation] += 1
                self.remove_bad_mutations(mutations_to_apply, available_mutations, allow_same_operators=False)
            yield mutations_to_apply


class RandomHOMStrategy(HOMStrategy):
    name = 'RANDOM'

    def __init__(self, *args, shuffler=random.shuffle, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffler = shuffler

    def generate(self, mutations):
        mutations = mutations[:]
        self.shuffler(mutations)
        while mutations:
            mutations_to_apply = []
            available_mutations = mutations[:]
            while len(mutations_to_apply) < self.order and available_mutations:
                try:
                    mutation = available_mutations.pop(0)
                    mutations_to_apply.append(mutation)
                    mutations.remove(mutation)
                except IndexError:
                    break
                self.remove_bad_mutations(mutations_to_apply, available_mutations)
            yield mutations_to_apply


hom_strategies = [
    BetweenOperatorsHOMStrategy,
    EachChoiceHOMStrategy,
    FirstToLastHOMStrategy,
    RandomHOMStrategy,
]


class FirstOrderMutator:

    def __init__(self, operators, percentage=100):
        self.operators = operators
        self.sampler = utils.RandomSampler(percentage)

    def mutate(self, target_ast, to_mutate=None, coverage_injector=None, module=None):
        for op in utils.sort_operators(self.operators):
            for mutation, mutant in op().mutate(target_ast, to_mutate, self.sampler, coverage_injector, module=module):
                yield [mutation], mutant


class HighOrderMutator(FirstOrderMutator):

    def __init__(self, *args, hom_strategy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hom_strategy = hom_strategy or FirstToLastHOMStrategy(order=2)

    def mutate(self, target_ast, to_mutate=None, coverage_injector=None, module=None):
        mutations = self.generate_all_mutations(coverage_injector, module, target_ast, to_mutate)
        for mutations_to_apply in self.hom_strategy.generate(mutations):
            generators = []
            applied_mutations = []
            mutant = target_ast
            for mutation in mutations_to_apply:
                generator = mutation.operator().mutate(
                    mutant,
                    to_mutate=to_mutate,
                    sampler=self.sampler,
                    coverage_injector=coverage_injector,
                    module=module,
                    only_mutation=mutation,
                )
                try:
                    new_mutation, mutant = generator.__next__()
                except StopIteration:
                    assert False, 'no mutations!'
                applied_mutations.append(new_mutation)
                generators.append(generator)
            yield applied_mutations, mutant
            self.finish_generators(generators)

    def generate_all_mutations(self, coverage_injector, module, target_ast, to_mutate):
        mutations = []
        for op in utils.sort_operators(self.operators):
            for mutation, _ in op().mutate(target_ast, to_mutate, None, coverage_injector, module=module):
                mutations.append(mutation)
        return mutations

    def finish_generators(self, generators):
        for generator in reversed(generators):
            try:
                generator.__next__()
            except StopIteration:
                continue
            assert False, 'too many mutations!'

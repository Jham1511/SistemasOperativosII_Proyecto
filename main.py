import sys
from collections import deque, OrderedDict
import time
import argparse
import heapq

class MemorySimulator:
    def __init__(self, frame_count, replacement_policy):
        self.frame_count = frame_count
        self.replacement_policy = replacement_policy.lower()
        self.page_table = {}  
        self.frame_table = {}  
        self.dirty_pages = set()
        self.page_faults = 0
        self.disk_writes = 0
        self.replacements = 0
        self.hits = 0
        self.total_accesses = 0
        self.operation_stats = {'R': 0, 'W': 0}
        self.page_access_frequency = {}
        self.start_time = time.time()

        if self.replacement_policy == 'fifo':
            self.fifo_queue = deque()
        elif self.replacement_policy == 'lru':
            self.lru_cache = OrderedDict()
        elif self.replacement_policy == 'opt':
            self.opt_future_refs = {}   
            self.next_ref = {} 

    def load_trace_file(self, filename):
        references = []
        try:
            with open(filename, 'r') as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) == 2:
                        try:
                            address = int(parts[0], 16)
                            operation = parts[1]
                            page_num = address >> 12
                            references.append((page_num, operation))
                            self.page_access_frequency[page_num] = self.page_access_frequency.get(page_num, 0) + 1
                        except ValueError:
                            continue
            
            if self.replacement_policy == 'opt':
                self._preprocess_opt_references(references)
            
            return references
        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo {filename}")
            sys.exit(1)

    def _preprocess_opt_references(self, references):
        page_positions = {}
        for pos, (page_num, _) in enumerate(references):
            if page_num not in page_positions:
                page_positions[page_num] = []
            page_positions[page_num].append(pos)
        
        for page_num in page_positions:
            positions = page_positions[page_num]
            heapq.heapify(positions)
            self.opt_future_refs[page_num] = positions
        
        self.next_ref = {}
        for page_num in self.opt_future_refs:
            if self.opt_future_refs[page_num]:
                self.next_ref[page_num] = heapq.heappop(self.opt_future_refs[page_num])

    def simulate(self, references):
        for current_pos, (page_num, operation) in enumerate(references):
            self.total_accesses += 1
            self.operation_stats[operation] += 1
            
            if self.replacement_policy == 'opt':
                self._update_opt_next_ref(page_num, current_pos)
            
            if page_num in self.page_table:
                self.hits += 1
                frame_num = self.page_table[page_num]
                
                if self.replacement_policy == 'lru':
                    self.lru_cache.move_to_end(page_num)
                
                if operation == 'W':
                    self.frame_table[frame_num]['dirty'] = True
                    self.dirty_pages.add(page_num)
            else:
                self.page_faults += 1
                self._handle_page_fault(page_num, operation, current_pos)

    def _update_opt_next_ref(self, page_num, current_pos):
        if page_num in self.next_ref and current_pos == self.next_ref[page_num]:
            if page_num in self.opt_future_refs and self.opt_future_refs[page_num]:
                self.next_ref[page_num] = heapq.heappop(self.opt_future_refs[page_num])
            else:
                self.next_ref[page_num] = None

    def _handle_page_fault(self, page_num, operation, current_pos):
        if len(self.page_table) < self.frame_count:
            frame_num = len(self.page_table)
        else:
            self.replacements += 1
            frame_num = self._select_victim_frame(current_pos)
            victim_page = self.frame_table[frame_num]['page_num']
            
            if self.frame_table[frame_num]['dirty']:
                self.disk_writes += 1
            
            del self.page_table[victim_page]
            if victim_page in self.dirty_pages:
                self.dirty_pages.remove(victim_page)
            
            if self.replacement_policy == 'lru' and victim_page in self.lru_cache:
                del self.lru_cache[victim_page]
        
        self.page_table[page_num] = frame_num
        self.frame_table[frame_num] = {
            'page_num': page_num,
            'dirty': (operation == 'W')
        }
        
        if operation == 'W':
            self.dirty_pages.add(page_num)
        
        if self.replacement_policy == 'fifo':
            self.fifo_queue.append(page_num)
        elif self.replacement_policy == 'lru':
            self.lru_cache[page_num] = frame_num

    def _select_victim_frame(self, current_pos):
        if self.replacement_policy == 'fifo':
            victim_page = self.fifo_queue.popleft()
            return self.page_table[victim_page]
        
        elif self.replacement_policy == 'lru':
            victim_page, _ = self.lru_cache.popitem(last=False)
            return self.page_table[victim_page]
        
        elif self.replacement_policy == 'opt':
            victim_page = None
            farthest_next_use = -1
            
            for page in self.page_table:
                next_use = self.next_ref.get(page, None)
                
                if next_use is None:
                    return self.page_table[page]
                
                if next_use > farthest_next_use:
                    farthest_next_use = next_use
                    victim_page = page
            
            return self.page_table[victim_page] if victim_page is not None else 0
        
        return 0

    def get_stats(self):
        if self.total_accesses == 0:
            return {}
        
        hit_rate = (self.hits / self.total_accesses) * 100
        fault_rate = (self.page_faults / self.total_accesses) * 100
        execution_time = time.time() - self.start_time
        eat = 100 + (fault_rate / 100 * 10_000_000)
        
        return {
            'total_accesses': self.total_accesses,
            'hits': self.hits,
            'page_faults': self.page_faults,
            'replacements': self.replacements,
            'disk_writes': self.disk_writes,
            'hit_rate': hit_rate,
            'fault_rate': fault_rate,
            'eat': eat,
            'execution_time': execution_time,
            'reads': self.operation_stats['R'],
            'writes': self.operation_stats['W'],
            'frames': self.frame_count,
            'policy': self.replacement_policy.upper(),
            'trace_file': getattr(self, 'trace_file', ''),
            'top_pages': sorted(self.page_access_frequency.items(), key=lambda x: x[1], reverse=True)[:20]
        }

def print_individual_report(result):
    print(f"\n{'='*80}")
    print(f"REPORTE INDIVIDUAL - {result['policy']} con {result['frames']} marcos")
    print(f"Archivo: {result['trace_file']}")
    print(f"{'='*80}")
    print("\nESTADÍSTICAS PRINCIPALES:")
    print(f"- Total accesos: {result['total_accesses']:,}")
    print(f"- Hits: {result['hits']:,} ({result['hit_rate']:.2f}%)")
    print(f"- Page faults: {result['page_faults']:,} ({result['fault_rate']:.2f}%)")
    print(f"- Reemplazos: {result['replacements']:,}")
    print(f"- Escrituras a disco: {result['disk_writes']:,}")
    print(f"- Tiempo de acceso efectivo (EAT): {result['eat']:,.2f} ns")
    print(f"- Tiempo de ejecución: {result['execution_time']:.2f} segundos")
    print("\nDISTRIBUCIÓN DE OPERACIONES:")
    total_ops = result['reads'] + result['writes']
    if total_ops > 0:
        read_pct = (result['reads'] / total_ops) * 100
        write_pct = (result['writes'] / total_ops) * 100
        print(f"- Lecturas (R): {result['reads']:,} ({read_pct:.2f}%)")
        print(f"- Escrituras (W): {result['writes']:,} ({write_pct:.2f}%)")
    print("\nTOP 10 PÁGINAS MÁS ACCEDIDAS:")
    print(f"{'Página':<15} {'Accesos':>12} {'% del total':>15}")
    print("-"*42)
    for page, count in result['top_pages'][:10]:
        pct = (count / result['total_accesses']) * 100
        print(f"0x{page:08X}{'':<7} {count:>12,} {pct:>14.2f}%")
    print(f"{'='*80}\n")

def run_simulations(trace_files, frame_counts, policies):
    all_results = []
    
    for trace_file in trace_files:
        print(f"\nProcesando archivo: {trace_file}")
        
        try:
            with open(trace_file, 'r') as f:
                total_lines = sum(1 for _ in f)
            print(f"Total referencias: {total_lines:,}")
        except FileNotFoundError:
            print("Error: Archivo no encontrado")
            continue
        
        for frames in frame_counts:
            for policy in policies:
                print(f"\nEjecutando {policy.upper()} con {frames} marcos...", end='', flush=True)
                
                simulator = MemorySimulator(frames, policy)
                simulator.trace_file = trace_file
                references = simulator.load_trace_file(trace_file)
                
                simulator.simulate(references)
                stats = simulator.get_stats()
                all_results.append(stats)
                
                print(" Completado")
    
    return all_results

def print_final_report(results):
    if not results:
        print("No hay resultados para mostrar.")
        return
    
    print("\n\n=== REPORTES INDIVIDUALES POR CONFIGURACIÓN ===")
    for res in results:
        print_individual_report(res)
    
    print("\n\n=== REPORTE COMPARATIVO FINAL ===")
    print("\nTabla Comparativa:")
    print("="*120)
    print(f"{'Archivo':<15} {'Marcos':>8} {'Política':>10} {'Accesos':>12} {'Hits':>10} {'Page Faults':>12} "
          f"{'Reemplazos':>12} {'Escrituras':>12} {'Hit Rate':>10} {'EAT (ns)':>12} {'Tiempo (s)':>10}")
    print("="*120)
    
    for res in results:
        print(f"{res['trace_file'][:15]:<15} {res['frames']:>8} {res['policy']:>10} "
              f"{res['total_accesses']:>12,} {res['hits']:>10,} {res['page_faults']:>12,} "
              f"{res['replacements']:>12,} {res['disk_writes']:>12,} {res['hit_rate']:>9.2f}% "
              f"{res['eat']:>12.2f} {res['execution_time']:>10.2f}")
    
    print("\n\nPáginas más accedidas (combinado):")
    print("="*60)
    print(f"{'Página':<15} {'Accesos':>12} {'% del total':>15}")
    print("="*60)
    
    combined_accesses = {}
    total_all_accesses = sum(r['total_accesses'] for r in results)
    
    for res in results:
        for page, count in res['top_pages']:
            combined_accesses[page] = combined_accesses.get(page, 0) + count
    
    top_combined = sorted(combined_accesses.items(), key=lambda x: x[1], reverse=True)[:20]
    
    for page, count in top_combined:
        percentage = (count / total_all_accesses) * 100
        print(f"0x{page:08X}{'':<7} {count:>12,} {percentage:>14.2f}%")

    print("\n\nComparación de Políticas:")
    print("="*90)
    print(f"{'Política':<10} {'Avg Hit Rate':>15} {'Avg Page Faults':>18} {'Avg EAT (ns)':>15} "
          f"{'Avg Escrituras':>15} {'Avg Tiempo (s)':>15}")
    print("="*90)
    
    policy_stats = {}
    for res in results:
        policy = res['policy']
        if policy not in policy_stats:
            policy_stats[policy] = {
                'hit_rates': [],
                'faults': [],
                'eats': [],
                'writes': [],
                'times': []
            }
        
        policy_stats[policy]['hit_rates'].append(res['hit_rate'])
        policy_stats[policy]['faults'].append(res['page_faults'])
        policy_stats[policy]['eats'].append(res['eat'])
        policy_stats[policy]['writes'].append(res['disk_writes'])
        policy_stats[policy]['times'].append(res['execution_time'])
    
    for policy, stats in policy_stats.items():
        avg_hit = sum(stats['hit_rates']) / len(stats['hit_rates'])
        avg_faults = sum(stats['faults']) / len(stats['faults'])
        avg_eat = sum(stats['eats']) / len(stats['eats'])
        avg_writes = sum(stats['writes']) / len(stats['writes'])
        avg_time = sum(stats['times']) / len(stats['times'])
        
        print(f"{policy:<10} {avg_hit:>14.2f}% {avg_faults:>17,.2f} {avg_eat:>14,.2f} "
              f"{avg_writes:>14,.2f} {avg_time:>14,.2f}")

def main():
    parser = argparse.ArgumentParser(description='Simulador de Memoria Virtual con Reporte Consolidado')
    parser.add_argument('trace_files', nargs='+', help='Archivo(s) de traza a procesar')
    parser.add_argument('--frames', nargs='+', type=int, default=[10, 50, 100],
                       help='Números de marcos a simular (default: 10 50 100)')
    parser.add_argument('--policies', nargs='+', default=['fifo', 'lru', 'opt'],
                       help='Políticas de reemplazo a evaluar (default: fifo lru opt)')
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    print("\nIniciando simulaciones...")
    results = run_simulations(args.trace_files, args.frames, args.policies)
    
    if results:
        print_final_report(results)
    else:
        print("\nNo se generaron resultados. Verifique los archivos de entrada.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\nError inesperado: {str(e)}")
        sys.exit(1)
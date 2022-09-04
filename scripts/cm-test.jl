using Distributed
using ClusterManagers
pmap(x -> x^2, 1:10)

em = ElasticManager(addr=:auto, port=0)
ElasticManager(cookie="foobar")


addprocs_pbs(5, qsub_flags=`-A open -l walltime=01:00:00 -l nodes=2:ppn=1 -l pmem=2gb`, wd=`/storage/home/suv87/work/julia/dp-rdpg`)















run(`/storage/work/s/suv87/julia_depot/juliaup/julia-1.7.3+0.x64/bin/julia --project=/storage/work/s/suv87/julia/dp-rdpg/Project.toml -e 'using ClusterManagers; ClusterManagers.elastic_worker("foobar","127.0.0.1",9009)'`)

while nworkers() != 5
    sleep(1)
end

pmap(x -> x^2, 1:10)

run(`qsub -A drh20_a_g_sc_default -l walltime=01:00:00 -l nodes=10:ppn=4 -l pmem=4gb "echo \"using ClusterManagers; ClusterManagers.elastic_worker(\"foobar\",\"hpc05\",port=9000) > julia"`)

addprocs(SlurmManager(2), partition="debug", t="00:5:00")

addprocs_pbs(5, qsub_flags=`-A drh20_a_g_sc_default -l walltime=01:00:00 -l nodes=4:ppn=4 -l pmem=4gb`)

addprocs_pbs(2, queue="drh20_a_g_sc_default")
